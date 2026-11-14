
from pathlib import Path

from src.validation.citations import (
    count_sentences,
    count_words,
    normalize_text,
)


def test_normalize_text():
    assert normalize_text("Hello   WORLD") == "hello world"


def test_count_words():
    assert count_words("This is an original speaker turn.") == 6


def test_count_sentences():
    assert count_sentences("First sentence. Second sentence!") == 2
    assert count_sentences("Only one sentence.") == 1
    assert count_sentences("One. Two. Three. Four.") == 4
