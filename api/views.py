import json
import traceback
from pathlib import Path

from django.core.exceptions import ValidationError

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .gemini_client import get_gemini_response
from .jenjo_prompt import get_system_prompt
from .models import Conversation, Message

# Load dataset once at module level
_DATASET_PATH = Path(__file__).resolve().parent.parent / 'data' / 'dza_language_dataset_v17.json'

try:
    with open(_DATASET_PATH, encoding='utf-8') as f:
        JENJO_DATA = json.load(f)
except FileNotFoundError:
    JENJO_DATA = {}


def _get_conversation(conversation_id):
    """Return an existing Conversation for the given id, or None. Tolerates a
    missing or malformed id (bad UUID) without raising."""
    if not conversation_id:
        return None
    try:
        return Conversation.objects.filter(id=conversation_id).first()
    except (ValueError, ValidationError):
        return None


@api_view(['POST'])
def chat(request):
    message = request.data.get('message', '').strip()
    conversation_id = request.data.get('conversation_id')
    regenerate = bool(request.data.get('regenerate', False))

    if not message:
        return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Resume the existing thread, or start a new one.
    conversation = _get_conversation(conversation_id) or Conversation.objects.create()

    # Retry/regenerate: drop the previous turn so this reply replaces it
    # (instead of stacking a duplicate).
    if regenerate:
        last_assistant = conversation.messages.filter(role='assistant').order_by('-created_at').first()
        last_user = conversation.messages.filter(role='user').order_by('-created_at').first()
        if last_assistant:
            last_assistant.delete()
        if last_user:
            last_user.delete()

    # History is sourced from the DB (the source of truth), not the client.
    history = [
        {'role': m.role, 'content': m.content}
        for m in conversation.messages.all()
    ]

    try:
        system_prompt = get_system_prompt(message)
        reply = get_gemini_response(system_prompt, history, message)
    except Exception as e:
        tb = traceback.format_exc()
        print("CHAT ERROR:\n", tb)
        msg = str(e)
        if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
            return Response(
                {'error': "I'm getting too many requests right now (Gemini free-tier limit). "
                          "Please wait about a minute and try again."},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        return Response({'error': str(e), 'detail': tb}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Persist the turn only after a successful reply, so a failed request leaves
    # no orphaned user message and a retry stays clean.
    Message.objects.create(conversation=conversation, role='user', content=message)
    Message.objects.create(conversation=conversation, role='assistant', content=reply)
    conversation.save(update_fields=['updated_at'])

    return Response({'reply': reply, 'conversation_id': str(conversation.id)})


def _conversation_title(conversation):
    """Use the first user message as the thread title, like Claude's sidebar."""
    first = conversation.messages.filter(role='user').first()
    if not first:
        return 'New conversation'
    text = first.content.strip().replace('\n', ' ')
    return text[:48] + ('…' if len(text) > 48 else '')


@api_view(['GET'])
def conversations_list(request):
    """List saved conversations (most recent first) for the sidebar."""
    items = [
        {
            'id': str(c.id),
            'title': _conversation_title(c),
            'updated_at': c.updated_at.isoformat(),
        }
        for c in Conversation.objects.all()
        if c.messages.exists()  # skip empty threads
    ]
    return Response(items)


@api_view(['GET', 'DELETE'])
def conversation_detail(request, conversation_id):
    """GET: return all messages so the frontend can resume the thread after a
    reload (unknown id returns an empty thread rather than 404).
    DELETE: remove the conversation and its messages."""
    conversation = _get_conversation(conversation_id)

    if request.method == 'DELETE':
        if conversation is not None:
            conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    if conversation is None:
        return Response({'conversation_id': None, 'messages': []})
    messages = [
        {'role': m.role, 'content': m.content}
        for m in conversation.messages.all()
    ]
    return Response({'conversation_id': str(conversation.id), 'messages': messages})


@api_view(['GET'])
def alphabet(request):
    return Response(JENJO_DATA.get('alphabet', []))


@api_view(['GET'])
def vocabulary(request):
    category = request.query_params.get('category', None)
    vocab = JENJO_DATA.get('vocabulary', {})
    if category and category in vocab:
        return Response(vocab[category])
    return Response(vocab)


@api_view(['GET'])
def phrases(request):
    vocab = JENJO_DATA.get('vocabulary', {})
    return Response(vocab.get('key_phrases', []))


@api_view(['GET'])
def numbers(request):
    return Response(JENJO_DATA.get('numbers', {}))
