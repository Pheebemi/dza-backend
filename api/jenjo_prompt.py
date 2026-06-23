import json
import re
import unicodedata
from pathlib import Path

_DATASET_PATH = Path(__file__).resolve().parent.parent / 'data' / 'dza_language_dataset_v17.json'

_BASE_PROMPT = None   # instructions + always-included core sections (cached)
_INDEX = None         # list of {dza, english, category, tokens} for retrieval (cached)

_INSTRUCTIONS = """You are a Jenjo (Dza) language tutor AI called "Pheebemi". You help users learn and communicate in the Jenjo/Dza language spoken in Taraba State, Nigeria by ~100,000 people.

RULES:
1. When user writes in English, translate to Jenjo and explain each word
2. When user writes in Jenjo, translate to English and explain grammar
3. Always show both languages
4. Be warm, encouraging, and conversational
5. If you don't know a word, say "I don't know this word yet in Jenjo - can you teach me?"
6. Use ONLY the vocabulary and grammar data provided - do not make up Jenjo words
7. For greetings, you can say "Səko!" (Hello/Greetings)
8. Refer to God as "Fi" and Jesus as "Yeso" when relevant
9. Keep responses concise but helpful
10. When teaching, break down sentences word by word

Note: a focused slice of the most relevant vocabulary is provided for each
question (the full dictionary is large). If you genuinely can't find a word in
what's provided, follow rule 5 rather than inventing one."""

# Words in a question that should pull in a whole vocabulary category.
_CATEGORY_ALIASES = {
    'animal': 'animals', 'animals': 'animals',
    'color': 'colors', 'colour': 'colors', 'colors': 'colors',
    'body': 'body_parts',
    'food': 'food_materials', 'eat': 'food_materials', 'drink': 'food_materials',
    'verb': 'verbs', 'verbs': 'verbs', 'action': 'verbs',
    'adjective': 'adjectives', 'adjectives': 'adjectives',
    'family': 'people_family', 'people': 'people_family', 'person': 'people_family',
    'nature': 'nature',
    'place': 'places_objects', 'places': 'places_objects', 'object': 'places_objects',
    'religious': 'religious_spiritual', 'god': 'religious_spiritual',
    'prayer': 'religious_spiritual', 'jesus': 'religious_spiritual',
    'phrase': 'key_phrases', 'phrases': 'key_phrases', 'greeting': 'key_phrases',
    'sentence': 'example_sentences', 'sentences': 'example_sentences',
}

_PUNCT = ' \t\n.,;:!?"\'“”‘’«»()[]{}<>—–-…/\\|@#*'

# Common English words to ignore when scoring relevance, so "how do I say water"
# locks onto "water" rather than every phrase containing "say"/"I".
_STOPWORDS = {
    'how', 'do', 'does', 'did', 'i', 'you', 'we', 'they', 'he', 'she', 'it',
    'say', 'said', 'tell', 'mean', 'means', 'what', 'whats', 'which', 'who',
    'the', 'a', 'an', 'to', 'of', 'in', 'on', 'is', 'are', 'am', 'be', 'and',
    'or', 'me', 'my', 'your', 'can', 'could', 'would', 'will', 'this', 'that',
    'for', 'with', 'about', 'word', 'words', 'jenjo', 'dza', 'english',
    'translate', 'translation', 'teach', 'learn', 'some', 'please', 'thanks',
    'hello', 'hi', 'name', 'call', 'called', 'use', 'used', 'using',
}


def _norm(s: str) -> str:
    return unicodedata.normalize('NFC', s)


def _tokens(s: str):
    out = []
    for w in re.split(r'[\s/]+', _norm(s)):
        w = w.strip(_PUNCT).lower()
        if w and not w.isdigit():
            out.append(w)
    return out


def _load():
    """Build and cache the base prompt (instructions + core sections) and the
    vocabulary retrieval index."""
    global _BASE_PROMPT, _INDEX
    if _BASE_PROMPT is not None:
        return

    try:
        with open(_DATASET_PATH, encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        _BASE_PROMPT = _INSTRUCTIONS + '\n\nERROR: dataset not found.'
        _INDEX = []
        return

    # Core (always included): everything except the big vocabulary dict. These
    # are small and essential (grammar, pronouns, numbers, alphabet, tones,
    # Lord's Prayer, etc.).
    core = {k: v for k, v in dataset.items() if k != 'vocabulary'}
    core_text = json.dumps(core, ensure_ascii=False, separators=(',', ':'))
    _BASE_PROMPT = (
        f'{_INSTRUCTIONS}\n\n'
        f'# CORE REFERENCE (grammar, pronouns, numbers, alphabet, tones, Lord\'s Prayer, etc.):\n'
        f'{core_text}'
    )

    # Vocabulary index for retrieval.
    index = []
    for category, entries in dataset.get('vocabulary', {}).items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            dza = e.get('dza', '')
            eng = e.get('english') or e.get('meaning') or ''
            index.append({
                'dza': dza,
                'english': eng,
                'category': category,
                'tokens': set(_tokens(dza)) | set(_tokens(eng)),
            })
    _INDEX = index


def _format_entry(e) -> str:
    return f'{e["dza"]} = {e["english"]}'


def _retrieve(user_message: str, max_entries: int = 80):
    """Return a list of vocabulary entries relevant to the user's message."""
    _load()
    q = {t for t in _tokens(user_message) if t not in _STOPWORDS}
    lower = _norm(user_message).lower()

    picked = []
    seen = set()

    def add(e):
        key = (e['category'], e['dza'])
        if key not in seen:
            seen.add(key)
            picked.append(e)

    # 1) Whole category if the user names one (e.g. "animals", "colors").
    wanted_categories = {cat for alias, cat in _CATEGORY_ALIASES.items() if alias in lower}
    if wanted_categories:
        for e in _INDEX:
            if e['category'] in wanted_categories:
                add(e)

    # 2) Keyword matches against both Jenjo and English (exact token, then substring).
    if q:
        scored = []
        for e in _INDEX:
            exact = len(q & e['tokens'])
            partial = 0
            if not exact:
                for qt in q:
                    if len(qt) >= 3 and any(qt in t for t in e['tokens']):
                        partial += 1
            score = exact * 2 + partial
            if score:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        for _, e in scored:
            add(e)
            if len(picked) >= max_entries:
                break

    # 3) Fallback: nothing matched -> give a useful default sample so the model
    #    still has material to work with (greetings/phrases + common words).
    if not picked:
        defaults = [e for e in _INDEX if e['category'] in ('key_phrases', 'function_words')]
        picked = defaults[:40]

    return picked[:max_entries]


def get_system_prompt(user_message: str = '') -> str:
    """Build the system prompt: always-included core + a focused slice of
    vocabulary relevant to this message (keeps token usage low)."""
    _load()
    entries = _retrieve(user_message) if user_message else []

    if entries:
        # group by category for readability
        by_cat = {}
        for e in entries:
            by_cat.setdefault(e['category'], []).append(e)
        lines = ['# RELEVANT VOCABULARY for this question (format: word = meaning):']
        for cat, items in by_cat.items():
            lines.append(f'\n## {cat}')
            lines.extend(_format_entry(e) for e in items)
        vocab_block = '\n'.join(lines)
        return f'{_BASE_PROMPT}\n\n{vocab_block}'

    return _BASE_PROMPT
