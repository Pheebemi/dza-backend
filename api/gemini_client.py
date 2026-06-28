import os

from google import genai
from google.genai import types

_MODEL = 'gemini-2.5-flash'
_CACHE_TTL = '3600s'  # re-cache the system prompt hourly

# Context caching is a PAID-tier feature (free tier storage limit = 0), so it's
# off by default. Set GEMINI_ENABLE_CACHE=1 once billing is enabled.
_CACHE_ENABLED = os.environ.get('GEMINI_ENABLE_CACHE', '').lower() in ('1', 'true', 'yes')

# Groq (optional): tried first (fast), Gemini as fallback. Supports a
# comma-separated list of keys (GROQ_API_KEYS), rotated round-robin and
# failed-over on rate limits, falling back to the single GROQ_API_KEY.
_GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
_GROQ_KEYS = [
    k.strip()
    for k in (os.environ.get('GROQ_API_KEYS') or os.environ.get('GROQ_API_KEY') or '').split(',')
    if k.strip()
]

_client = None
_groq_clients = {}      # api_key -> Groq client (created lazily)
_groq_rr = 0            # round-robin pointer across the keys
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


def _get_groq_client(key: str):
    """Lazily create (and cache) a Groq client for a given API key."""
    if key not in _groq_clients:
        from groq import Groq
        _groq_clients[key] = Groq(api_key=key)
    return _groq_clients[key]


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
    global _groq_rr
    if not _GROQ_KEYS:
        raise RuntimeError('Groq not configured')

    messages = [{'role': 'system', 'content': system_prompt}]
    for m in conversation_history:
        role = 'assistant' if m.get('role') == 'assistant' else 'user'
        messages.append({'role': role, 'content': m.get('content', '')})
    messages.append({'role': 'user', 'content': user_message})

    # Round-robin across keys; on failure (e.g. rate limit) try the next one.
    n = len(_GROQ_KEYS)
    start = _groq_rr
    _groq_rr = (_groq_rr + 1) % n
    last_err = None
    for offset in range(n):
        key = _GROQ_KEYS[(start + offset) % n]
        try:
            resp = _get_groq_client(key).chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                temperature=0.6,
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text
            last_err = RuntimeError('Groq returned empty response')
        except Exception as e:
            last_err = e
            print(f'Groq key #{(start + offset) % n + 1}/{n} failed, trying next:', e)
    raise last_err or RuntimeError('All Groq keys failed')


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
    if _GROQ_KEYS:
        try:
            return _groq_response(system_prompt, conversation_history, user_message)
        except Exception as e:
            print('Groq failed (all keys), falling back to Gemini:', e)

    return _gemini_response(system_prompt, conversation_history, user_message)
