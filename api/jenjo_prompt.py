import json
from pathlib import Path

_SYSTEM_PROMPT = None


def _serialize_dataset(dataset: dict) -> str:
    """Compact, token-efficient rendering of the dataset for the prompt.

    Vocabulary (the bulk) becomes plain `word = meaning` lines per category
    instead of repeating JSON keys ~1900x. Everything else (grammar, pronouns,
    numbers, alphabet, phrases, etc.) stays as minified JSON so no content is
    lost. ~34% fewer characters than dumping the whole thing as JSON, with the
    same information available to the model."""
    parts = ['# VOCABULARY (format: word = meaning)']
    for category, entries in dataset.get('vocabulary', {}).items():
        parts.append(f'\n## {category}')
        if isinstance(entries, list):
            for e in entries:
                if not isinstance(e, dict):
                    continue
                dza = e.get('dza', '')
                eng = e.get('english') or e.get('meaning') or ''
                line = f'{dza} = {eng}'
                if e.get('note'):
                    line += f' [{e["note"]}]'
                parts.append(line)
    vocab_text = '\n'.join(parts)

    # Keep all non-vocabulary sections, minified (they're small relative to vocab).
    other = {k: v for k, v in dataset.items() if k != 'vocabulary'}
    other_text = json.dumps(other, ensure_ascii=False, separators=(',', ':'))

    return (
        f'{vocab_text}\n\n'
        f'# GRAMMAR, PRONOUNS, NUMBERS, ALPHABET, TONES, LORD\'S PRAYER, ETC. (JSON):\n'
        f'{other_text}'
    )


def get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is not None:
        return _SYSTEM_PROMPT

    dataset_path = Path(__file__).resolve().parent.parent / 'data' / 'dza_language_dataset_v17.json'

    try:
        with open(dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)
        dataset_str = _serialize_dataset(dataset)
    except FileNotFoundError:
        dataset_str = 'ERROR: dataset not found. Please add dza_language_dataset_v17.json to backend/data/'

    _SYSTEM_PROMPT = f"""You are a Jenjo (Dza) language tutor AI called "Pheebemi". You help users learn and communicate in the Jenjo/Dza language spoken in Taraba State, Nigeria by ~100,000 people.

RULES:
1. When user writes in English, translate to Jenjo and explain each word
2. When user writes in Jenjo, translate to English and explain grammar
3. Always show both languages
4. Be warm, encouraging, and conversational
5. If you don't know a word, say "I don't know this word yet in Jenjo - can you teach me?"
6. Use ONLY the vocabulary and grammar data provided below - do not make up Jenjo words
7. For greetings, you can say "Səko!" (Hello/Greetings)
8. Refer to God as "Fi" and Jesus as "Yeso" when relevant
9. Keep responses concise but helpful
10. When teaching, break down sentences word by word

JENJO LANGUAGE DATA:
{dataset_str}"""

    return _SYSTEM_PROMPT
