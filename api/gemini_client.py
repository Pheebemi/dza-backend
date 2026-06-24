import os

from google import genai
from google.genai import types

_MODEL = 'gemini-2.5-flash'
_CACHE_TTL = '3600s'  # re-cache the system prompt hourly

# Context caching is a PAID-tier feature (free tier storage limit = 0), so it's
# off by default. Set GEMINI_ENABLE_CACHE=1 once billing is enabled.
_CACHE_ENABLED = os.environ.get('GEMINI_ENABLE_CACHE', '').lower() in ('1', 'true', 'yes')

# Groq (optional): if a key is present we try Groq first (fast, separate free
# quota) and fall back to Gemini on any failure.
_GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')

_client = None
_groq_client = None
_cache_name = None      # name of the cached system prompt, reused across requests
_caching_disabled = not _CACHE_ENABLED  # stop retrying once we know it won't work


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
    return _client


def _get_groq_client():
    """Return a Groq client if GROQ_API_KEY is set, else None."""
    global _groq_client
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        return None
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=key)
    return _groq_client


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


# --------------------------------------------------------------------------- #
# Groq (primary when available)
# --------------------------------------------------------------------------- #
def _groq_response(system_prompt: str, conversation_history: list, user_message: str) -> str:
    client = _get_groq_client()
    if client is None:
        raise RuntimeError('Groq not configured')

    messages = [{'role': 'system', 'content': system_prompt}]
    for m in conversation_history:
        role = 'assistant' if m.get('role') == 'assistant' else 'user'
        messages.append({'role': role, 'content': m.get('content', '')})
    messages.append({'role': 'user', 'content': user_message})

    resp = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=messages,
        temperature=0.6,
    )
    text = resp.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError('Groq returned empty response')
    return text


# --------------------------------------------------------------------------- #
# Gemini (fallback, or primary if no Groq key)
# --------------------------------------------------------------------------- #
def _get_cache(system_prompt: str):
    """Get-or-create a cached copy of the system prompt (paid tier only)."""
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
        print('Context caching unavailable, using inline prompt:', e)
        _cache_name = None
        _caching_disabled = True
    return _cache_name


def _gemini_response(system_prompt: str, conversation_history: list, user_message: str) -> str:
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
            print('Cached request failed, falling back to inline prompt:', e)
            _cache_name = None

    chat = client.chats.create(
        model=_MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        history=history,
    )
    return chat.send_message(user_message).text


# --------------------------------------------------------------------------- #
# Dispatcher: Groq primary -> Gemini fallback
# --------------------------------------------------------------------------- #
def get_gemini_response(system_prompt: str, conversation_history: list, user_message: str) -> str:
    """Try Groq first (if configured), then fall back to Gemini. Raises the
    last error only if BOTH providers fail."""
    if _get_groq_client() is not None:
        try:
            return _groq_response(system_prompt, conversation_history, user_message)
        except Exception as e:
            print('Groq failed, falling back to Gemini:', e)

    return _gemini_response(system_prompt, conversation_history, user_message)
