"""Step 2 - preprocessing: from raw narrative to clean, lemmatised tokens.

Sequence (as planned in the concept, with the order of two steps adjusted):

1. Remove what the CFPB inserts for privacy - XXXX redactions and {$amounts} - plus URLs,
   e-mail addresses, digits and punctuation (regular expressions).
2. Tokenise, tag parts of speech and lemmatise with spaCy, so that "payments", "paid" and
   "paying" all become "pay". Lowercasing happens after tagging, because spaCy uses
   capitalisation to recognise proper nouns.
3. Keep content words (nouns, proper nouns, verbs, adjectives) of at least three letters.
4. Remove stop words: NLTK's English list plus a short, documented list of words that are
   frequent in complaints but carry no topic (config.yaml). Removing them after
   lemmatisation catches every inflected form ("said", "says" -> "say").
5. Join frequent two-word expressions such as credit_report or identity_theft (gensim
   Phrases), so that the vocabulary contains unigrams and meaningful bigrams.
6. Discard documents with fewer than 20 tokens; they give the topic models too little to
   work with.
"""
from __future__ import annotations

import re
from collections import Counter

import nltk
import pandas as pd
import spacy
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases

from .utils import TABLES, load, log, save, timed

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_AMOUNT = re.compile(r"\{\$?[\d,.]*\}")             # {$1,200.00}
_REDACTION = re.compile(r"\b[xX]{2,}\b")            # XXXX, and XX/XX/XXXX once split
_NOT_LETTER = re.compile(r"[^A-Za-z]+")


def clean_text(text: str) -> str:
    """Step 1: strip redactions, amounts, links, digits and punctuation (case is kept)."""
    t = _URL.sub(" ", text)
    t = _EMAIL.sub(" ", t)
    t = _AMOUNT.sub(" ", t)
    t = _NOT_LETTER.sub(" ", t)          # also splits XX/XX/XXXX into XX XX XXXX
    t = _REDACTION.sub(" ", t)
    return " ".join(t.split())


def stopword_set(cfg: dict) -> set:
    try:
        words = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("english")
    return set(words) | set(cfg["preprocessing"]["domain_stopwords"])


def load_spacy(cfg: dict):
    # The parser and named-entity recogniser are not needed and would only cost time.
    return spacy.load(cfg["preprocessing"]["spacy_model"], disable=["parser", "ner"])


def tokens_from_doc(doc, stop: set, keep_pos: set, min_len: int) -> list:
    """Steps 2-4 for one spaCy document."""
    out = []
    for tok in doc:
        if tok.pos_ not in keep_pos:
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) < min_len or not lemma.isalpha():
            continue
        if lemma in stop or tok.lower_ in stop:
            continue
        out.append(lemma)
    return out


def tokenize(texts: list, cfg: dict, n_process: int | None = None) -> list:
    p = cfg["preprocessing"]
    nlp = load_spacy(cfg)
    stop = stopword_set(cfg)
    keep_pos = set(p["keep_pos"])
    docs = nlp.pipe(texts, batch_size=p["batch_size"],
                    n_process=p["n_process"] if n_process is None else n_process)
    return [tokens_from_doc(d, stop, keep_pos, p["min_token_length"]) for d in docs]


def add_phrases(token_lists: list, cfg: dict) -> tuple:
    """Step 5: detect frequent two-word expressions and join them with an underscore."""
    ph = cfg["preprocessing"]["phrases"]
    model = Phrases(token_lists, min_count=ph["min_count"], threshold=ph["threshold"],
                    connector_words=ENGLISH_CONNECTOR_WORDS)
    frozen = model.freeze()
    joined = [frozen[t] for t in token_lists]
    counts = Counter(w for t in joined for w in t if "_" in w)
    return joined, counts


def run(cfg: dict) -> None:
    sample = load("sample.pkl")
    unique = sample[~sample["is_duplicate"]].reset_index(drop=True)
    log.info("preprocessing %d unique narratives", len(unique))

    with timed("clean the raw text"):
        cleaned = [clean_text(t) for t in unique["narrative"]]
    with timed("tokenise, tag and lemmatise with spaCy"):
        tokens = tokenize(cleaned, cfg)
    with timed("detect two-word phrases"):
        tokens, phrase_counts = add_phrases(tokens, cfg)

    unique["tokens"] = tokens
    unique["n_tokens"] = unique["tokens"].str.len()
    min_tokens = cfg["preprocessing"]["min_doc_tokens"]
    corpus = unique[unique["n_tokens"] >= min_tokens].reset_index(drop=True)
    log.info("kept %d documents with at least %d tokens (%d too short)",
             len(corpus), min_tokens, len(unique) - len(corpus))

    cols = ["complaint_id", "date_received", "product", "sub_product", "issue", "company",
            "tokens", "n_tokens"]
    save(corpus[cols], "corpus.pkl")

    pd.DataFrame(phrase_counts.most_common(40), columns=["phrase", "count"]).to_csv(
        TABLES / "top_phrases.csv", index=False)
    lengths = corpus["n_tokens"].describe(percentiles=[0.1, 0.5, 0.9]).round(1)
    summary = pd.DataFrame({
        "unique_narratives": [len(unique)],
        "dropped_short": [len(unique) - len(corpus)],
        "documents": [len(corpus)],
        "tokens_total": [int(corpus["n_tokens"].sum())],
        "tokens_median": [lengths["50%"]],
        "tokens_p10": [lengths["10%"]],
        "tokens_p90": [lengths["90%"]],
        "distinct_tokens": [len({w for t in corpus["tokens"] for w in t})],
    })
    summary.to_csv(TABLES / "preprocessing_summary.csv", index=False)
