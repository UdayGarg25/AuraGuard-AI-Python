"""Covers the 4 required Phase 4 test scenarios. Gemini is mocked so these
run offline/deterministically without a real API key — see
scripts/test_speech_cli.py for exercising a real Gemini call."""

from unittest.mock import patch

from speech import keyword_filter, stt_service
from speech.pipeline import process_text
from speech.schemas import GeminiAnalysis


def test_normal_sentence_never_calls_gemini():
    """TEST 1 — normal sentence: no keyword match, Gemini must not be called."""
    with patch("speech.gemini_service.analyze_with_gemini") as mock_gemini:
        result = process_text("Hey, are we still meeting at 3pm today?")

    mock_gemini.assert_not_called()
    assert result.keyword_matched is False
    assert result.llm_analyzed is False
    assert result.flagged is False


def test_abusive_word_triggers_gemini():
    """TEST 2 — abusive word: manual check detects it, Gemini is called."""
    fake_gemini_result = GeminiAnalysis(
        flagged=True, type="abusive_language", severity="medium", confidence=0.75, reason="Insulting language directed at another participant."
    )
    with patch("speech.gemini_service.analyze_with_gemini", return_value=fake_gemini_result) as mock_gemini:
        result = process_text("You are such an idiot, shut up.")

    mock_gemini.assert_called_once()
    assert result.keyword_matched is True
    assert result.matched_category == "abusive_words"
    assert result.llm_analyzed is True
    assert result.flagged is True
    assert result.type == "abusive_language"


def test_threatening_sentence_flagged_by_gemini():
    """TEST 3 — threatening sentence: keyword matches, Gemini confirms threat."""
    fake_gemini_result = GeminiAnalysis(
        flagged=True, type="violence", severity="high", confidence=0.92, reason="The speaker appears to make a direct threat."
    )
    with patch("speech.gemini_service.analyze_with_gemini", return_value=fake_gemini_result) as mock_gemini:
        result = process_text("I will kill you if you do that again.")

    mock_gemini.assert_called_once()
    assert result.keyword_matched is True
    assert result.matched_category == "threatening_phrases"
    assert result.flagged is True
    assert result.severity == "high"
    assert result.type == "violence"


def test_suspicious_word_harmless_context_not_flagged():
    """TEST 4 — suspicious word, harmless context: keyword may trigger,
    Gemini determines it's not an actual violation."""
    fake_gemini_result = GeminiAnalysis(
        flagged=False, type=None, severity="none", confidence=0.90, reason="The phrase is being used in a non-threatening context."
    )
    with patch("speech.gemini_service.analyze_with_gemini", return_value=fake_gemini_result) as mock_gemini:
        result = process_text("That movie had such a violent scene, it made me jump.")

    mock_gemini.assert_called_once()
    assert result.keyword_matched is True
    assert result.llm_analyzed is True
    assert result.flagged is False


def test_keyword_filter_matches_phrase_and_single_word_categories():
    assert keyword_filter.check_text("I will kill you").category == "threatening_phrases"
    assert keyword_filter.check_text("get the gun").category == "violent_words"
    assert keyword_filter.check_text("nothing suspicious here at all").matched is False


def test_keyword_filter_does_not_match_substrings_of_unrelated_words():
    # "ass" must not match inside "class"/"pass" etc. — word-boundary check.
    result = keyword_filter.check_text("Let's pass the class assignment around")
    assert result.matched is False


def test_better_profanity_catches_words_not_in_the_custom_json():
    # "bullshit" is deliberately NOT in prohibited_words.json — this only
    # matches because better_profanity's built-in dictionary covers it.
    result = keyword_filter.check_text("This is such bullshit honestly")
    assert result.matched is True
    assert result.category == "profanity"
    assert "bullshit" in result.matched_terms


def test_profanity_triggers_gemini_via_pipeline():
    """A curse word outside the custom list still reaches Gemini through
    the better_profanity path, not just the hand-curated JSON."""
    fake_gemini_result = GeminiAnalysis(
        flagged=True, type="abusive_language", severity="low", confidence=0.6, reason="Mild profanity, not directed as a threat."
    )
    with patch("speech.gemini_service.analyze_with_gemini", return_value=fake_gemini_result) as mock_gemini:
        result = process_text("This is such bullshit honestly")

    mock_gemini.assert_called_once()
    assert result.matched_category == "profanity"
    assert result.llm_analyzed is True


def test_malformed_audio_returns_empty_text_instead_of_crashing():
    # A real network chunk could arrive truncated/corrupted — this must
    # degrade to "no speech detected", never a 500.
    result = stt_service.transcribe_audio_bytes(b"not-a-real-audio-file", suffix=".webm")
    assert result == ""
