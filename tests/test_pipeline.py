"""Unit tests for the parts of the pipeline whose behaviour is easiest to get wrong.

Run with:  python -m pytest -q
"""
import numpy as np

from pipeline.data import find_duplicates, normalise_for_dedup, sample_key
from pipeline.preprocess import clean_text, tokenize
from pipeline.utils import load_config

CFG = load_config()


def test_clean_text_removes_cfpb_redactions_amounts_and_links():
    raw = "On XX/XX/2024 XXXX charged me {$1,200.00}. See https://example.com or mail a@b.com!"
    assert clean_text(raw) == "On charged me See or mail"


def test_tokenize_lemmatises_and_removes_stop_words():
    tokens = tokenize(["The bank charged me late fees and kept calling my phone."], CFG,
                      n_process=1)[0]
    assert "charge" in tokens and "fee" in tokens       # lemmatised
    assert "the" not in tokens and "and" not in tokens  # NLTK stop words
    assert "call" not in tokens and "keep" not in tokens  # domain stop words
    assert all(len(t) >= 3 for t in tokens)


def test_sample_key_is_deterministic_and_seed_dependent():
    ids = ["101", "102", "103"]
    assert (sample_key(ids, 42) == sample_key(ids, 42)).all()
    assert not (sample_key(ids, 42) == sample_key(ids, 7)).all()


def test_normalise_for_dedup_ignores_redactions_and_amounts():
    a = normalise_for_dedup("Account XXXX was charged {$15.00} on XX/XX/2024.")
    b = normalise_for_dedup("Account XXXX XXXX was charged {$990.00} on XX/XX/2023.")
    assert a == b == "account was charged on"


def test_find_duplicates_groups_a_template_with_its_variant():
    texts = [
        "please remove the fraudulent account from my credit file immediately it is not mine",
        "please remove the fraudulent account from my credit file immediately it is not mine at all",
        "my mortgage servicer lost my payment",
        "the bank closed my checking account without notice",
        "a debt collector keeps calling about a debt i do not owe",
        "my student loan was transferred to a new company",
    ]
    rep = find_duplicates(texts, order=np.arange(len(texts)), threshold=0.9)
    assert rep[1] == 0                                   # variant -> template
    assert [rep[i] for i in (0, 2, 3, 4, 5)] == [0, 2, 3, 4, 5]  # the rest stay unique
