import json
import traceback
from pathlib import Path

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .gemini_client import get_gemini_response
from .jenjo_prompt import get_system_prompt

# Load dataset once at module level
_DATASET_PATH = Path(__file__).resolve().parent.parent / 'data' / 'dza_language_dataset_v6.json'

try:
    with open(_DATASET_PATH, encoding='utf-8') as f:
        JENJO_DATA = json.load(f)
except FileNotFoundError:
    JENJO_DATA = {}


@api_view(['POST'])
def chat(request):
    message = request.data.get('message', '').strip()
    history = request.data.get('history', [])

    if not message:
        return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        system_prompt = get_system_prompt()
        reply = get_gemini_response(system_prompt, history, message)
        return Response({'reply': reply})
    except Exception as e:
        return Response({'error': str(e), 'detail': traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
