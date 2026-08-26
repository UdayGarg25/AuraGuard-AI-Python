"""Local/manual bad-word & violent-phrase filter — Step 4 of the pipeline.

This is a lightweight FIRST-LEVEL trigger, not a moderation decision. Its
only job is: "is this text suspicious enough to spend a Gemini call on?"

Two complementary sources, deliberately kept separate:

- speech/data/prohibited_words.json — hand-curated THREAT/VIOLENCE/
  HARASSMENT phrases and mild non-swear insults. These are contextual
  phrases ("I will kill you"), not swear words, so no generic profanity
  dictionary would ever contain them. Editable without touching code.
- better_profanity's built-in dictionary — thousands of profanity/abusive
  words, maintained upstream, instead of hand-typing a tiny list that's
  always missing something.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from better_profanity import profanity

from speech.schemas import KeywordMatchResult

DATA_PATH = Path(__file__).parent / "data" / "prohibited_words.json"

# Checked in this order: multi-word phrases first (more specific, higher
# signal), then single words. Whichever category matches first wins for
# `matched_category`, but ALL matched terms are still reported. Custom
# threat/violence/harassment phrases outrank generic profanity matches.
CATEGORY_ORDER = [
    "threatening_phrases",
    "violent_words",
    "harassment_terms",
    "abusive_words",
]

PROFANITY_CATEGORY = "profanity"


@lru_cache
def _load_word_lists() -> dict[str, list[str]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {category: data[category] for category in CATEGORY_ORDER if category in data}


@lru_cache
def _profanity_engine():
    # Loaded once (lru_cache) — this pulls in better_profanity's full
    # built-in wordlist (thousands of entries), not a hand-typed subset.
    profanity.load_censor_words()
    return profanity


def _contains_term(text_lower: str, term: str) -> bool:
    # Word-boundary match for single words (so "class" doesn't match "ass"),
    # plain substring match for multi-word phrases (word boundaries on both
    # ends of a phrase are unnecessary and would miss punctuation variants).
    if " " in term:
        return term.lower() in text_lower
    return re.search(rf"\b{re.escape(term.lower())}\b", text_lower) is not None


def _profanity_matches(text: str) -> list[str]:
    """Returns the specific word(s) better_profanity flagged, by diffing the
    original text against its censored version (censor() replaces each
    matched word with asterisks; contains_profanity() alone doesn't say
    which word matched)."""
    engine = _profanity_engine()
    if not engine.contains_profanity(text):
        return []

    original_words = text.split()
    censored_words = engine.censor(text).split()

    return [
        original.strip(".,!?;:\"'").lower()
        for original, censored in zip(original_words, censored_words)
        if original != censored and set(censored) <= {"*"}
    ]


def check_text(text: str) -> KeywordMatchResult:
    """Scans `text` against the custom phrase lists and the full
    better_profanity dictionary. Returns the highest-priority matched
    category plus every term that matched, so the caller has full
    visibility."""

    if not text or not text.strip():
        return KeywordMatchResult(matched=False)

    text_lower = text.lower()
    word_lists = _load_word_lists()

    matched_category: str | None = None
    matched_terms: list[str] = []

    for category in CATEGORY_ORDER:
        for term in word_lists.get(category, []):
            if _contains_term(text_lower, term):
                matched_terms.append(term)
                if matched_category is None:
                    matched_category = category

    profanity_terms = _profanity_matches(text)
    if profanity_terms:
        matched_terms.extend(profanity_terms)
        if matched_category is None:
            matched_category = PROFANITY_CATEGORY

    return KeywordMatchResult(
        matched=matched_category is not None,
        category=matched_category,
        matched_terms=matched_terms,
    )
