import os

from google import genai
from google.genai import types

_MODEL = 'gemini-2.5-flash'
_CACHE_TTL = '3600s'  # re-cache the system prompt hourly

# Context caching is a PAID-tier feature (free tier storage limit = 0), so it's
# off by default. Set GEMINI_ENABLE_CACHE=1 once billing is enabled.
_CACHE_ENABLED = os.environ.get('GEMINI_ENABLE_CACHE', '').lower() in ('1', 'true', 'yes')

_client = None
_cache_name = None      # name of the cached system prompt, reused across requests
_caching_disabled = not _CACHE_ENABLED  # stop retrying once we know it won't work


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
    return _client


def _get_cache(system_prompt: str):
    """Get-or-create a cached copy of the (large, unchanging) system prompt so
    we don't resend ~28k tokens on every message. Returns the cache name, or
    None if caching is unavailable (then we fall back to inline)."""
    global _cache_name, _caching_disabled
    if _caching_disabled:
        return None
    if _cache_name is not None:
        return _cache_name
    try:
        cache = _get_client().caches.create(
            model=_MODEL,
            config=types.CreateCachedContentConfig(
                display_name='jenjo-pheebemi-system',
                system_instruction=system_prompt,
                ttl=_CACHE_TTL,
            ),
        )
        _cache_name = cache.name
    except Exception as e:
        # e.g. free tier (storage limit 0) or below caching minimum. Don't keep
        # retrying for the rest of this process — just use the inline prompt.
        print('Context caching unavailable, using inline prompt:', e)
        _cache_name = None
        _caching_disabled = True
    return _cache_name


def _format_history(conversation_history: list):
    formatted = []
    for msg in conversation_history:
        role = msg.get('role', 'user')
        if role == 'assistant':
            role = 'model'
        formatted.append(
            types.Content(role=role, parts=[types.Part(text=msg.get('content', ''))])
        )
    return formatted


def get_gemini_response(system_prompt: str, conversation_history: list, user_message: str) -> str:
    global _cache_name
    client = _get_client()
    history = _format_history(conversation_history)

    cache_name = _get_cache(system_prompt)
    if cache_name:
        try:
            chat = client.chats.create(
                model=_MODEL,
                config=types.GenerateContentConfig(cached_content=cache_name),
                history=history,
            )
            return chat.send_message(user_message).text
        except Exception as e:
            # Cache likely expired/evicted — drop it and fall back; it will be
            # recreated on the next request.
            print('Cached request failed, falling back to inline prompt:', e)
            _cache_name = None

    chat = client.chats.create(
        model=_MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        history=history,
    )
    return chat.send_message(user_message).text
