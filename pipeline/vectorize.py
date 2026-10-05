"""Step 3 - vectorisation: two ways of turning token lists into numbers, and a comparison.

1. Bag of words / TF-IDF. Every document becomes a sparse vector with one entry per
   vocabulary term. LDA works on the raw counts (bag of words), because it models how
   often words occur; NMF works on TF-IDF weights, which down-weight words that occur in
   many documents. Both use the same vocabulary.
2. Word2Vec. A skip-gram model learns a dense 100-dimensional vector per word from the
   contexts it appears in, so related words ("loan", "mortgage") get similar vectors. A
   document is represented by the average of its word vectors.

The comparison asks a concrete question: are a complaint's nearest neighbours about the
same CFPB product? The product label is never used to build either representation, so
agreement with it shows how much topical information each one captures.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .utils import TABLES, load, log, save, timed

SHORT_PRODUCT = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card": "Credit card",
    "Checking or savings account": "Checking/savings",
    "Mortgage": "Mortgage",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Prepaid card": "Prepaid card",
    "Debt or credit management": "Debt management",
}


def short_product(p: str) -> str:
    return SHORT_PRODUCT.get(p, p)


def _identity(doc):
    return doc


def _stable_hash(s: str) -> int:
    # Python's built-in hash() changes between runs; this one does not, which makes the
    # Word2Vec initialisation reproducible.
    return zlib.crc32(s.encode("utf-8"))


def build_vocabulary(tokens: list, cfg: dict) -> Dictionary:
    v = cfg["vectorization"]
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=v["no_below"], no_above=v["no_above"],
                               keep_n=v["keep_n"])
    dictionary.compactify()
    return dictionary


def build_tfidf(tokens: list, vocab: list):
    vec = TfidfVectorizer(analyzer=_identity, vocabulary=vocab)
    return vec.fit_transform(tokens), vec


def train_word2vec(tokens: list, cfg: dict) -> Word2Vec:
    w = cfg["vectorization"]["word2vec"]
    return Word2Vec(sentences=tokens, vector_size=w["vector_size"], window=w["window"],
                    min_count=w["min_count"], sg=w["sg"], epochs=w["epochs"],
                    seed=cfg["seed"], workers=1, hashfxn=_stable_hash)


def document_vectors(tokens: list, wv) -> np.ndarray:
    out = np.zeros((len(tokens), wv.vector_size), dtype=np.float32)
    for i, doc in enumerate(tokens):
        known = [w for w in doc if w in wv.key_to_index]
        if known:
            out[i] = wv[known].mean(axis=0)
    return out


def neighbour_agreement(X, labels: np.ndarray, k: int) -> pd.Series:
    """Share of each document's k nearest neighbours (cosine) that have its label."""
    Xn = normalize(X)
    sim = Xn @ Xn.T
    sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    same = (labels[nn] == labels[:, None]).mean(axis=1)
    return pd.Series(same).groupby(labels).mean()


def run(cfg: dict) -> None:
    corpus = load("corpus.pkl")
    tokens = corpus["tokens"].tolist()
    v = cfg["vectorization"]

    with timed("build the shared vocabulary"):
        dictionary = build_vocabulary(tokens, cfg)
    vocab = [dictionary[i] for i in range(len(dictionary))]
    log.info("vocabulary: %d terms", len(vocab))

    t0 = time.time()
    bow = [dictionary.doc2bow(t) for t in tokens]
    tfidf, _ = build_tfidf(tokens, vocab)
    t_tfidf = time.time() - t0

    with timed("train Word2Vec"):
        t0 = time.time()
        w2v = train_word2vec(tokens, cfg)
        doc_vecs = document_vectors(tokens, w2v.wv)
        t_w2v = time.time() - t0

    save({"dictionary": dictionary, "vocab": vocab, "bow": bow, "tfidf": tfidf}, "bow_tfidf.pkl")
    save({"wv": w2v.wv, "doc_vectors": doc_vecs}, "word2vec.pkl")

    # ---- comparison 1: basic properties
    props = pd.DataFrame([
        {"representation": "TF-IDF (bag of words)", "dimensions": tfidf.shape[1],
         "non_zero_share": round(tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]), 5),
         "memory_mb": round((tfidf.data.nbytes + tfidf.indices.nbytes + tfidf.indptr.nbytes) / 1e6, 1),
         "build_seconds": round(t_tfidf, 1), "interpretable_dimensions": "yes (one per term)"},
        {"representation": "Word2Vec (mean of word vectors)", "dimensions": doc_vecs.shape[1],
         "non_zero_share": round(float((doc_vecs != 0).mean()), 5),
         "memory_mb": round(doc_vecs.nbytes / 1e6, 1),
         "build_seconds": round(t_w2v, 1), "interpretable_dimensions": "no"},
    ])

    # ---- comparison 2: what does Word2Vec consider similar?
    rows = []
    for w in v["comparison"]["probe_words"]:
        if w in w2v.wv.key_to_index:
            sims = w2v.wv.most_similar(w, topn=6)
            rows.append({"word": w, "nearest_words": ", ".join(f"{s} ({c:.2f})" for s, c in sims)})
    pd.DataFrame(rows).to_csv(TABLES / "word2vec_neighbours.csv", index=False)

    # ---- comparison 3: neighbour agreement with the product label (balanced subset)
    rng = np.random.default_rng(cfg["seed"])
    per = v["comparison"]["eval_docs_per_product"]
    counts = corpus["product"].value_counts()
    eval_idx = np.concatenate([
        rng.choice(np.flatnonzero(corpus["product"].to_numpy() == p), per, replace=False)
        for p in counts[counts >= per].index])
    labels = corpus["product"].to_numpy()[eval_idx]
    k = v["comparison"]["neighbours"]
    agree_tfidf = neighbour_agreement(tfidf[eval_idx], labels, k)
    agree_w2v = neighbour_agreement(doc_vecs[eval_idx], labels, k)
    agreement = pd.DataFrame({"tfidf": agree_tfidf, "word2vec": agree_w2v}).round(3)
    agreement.index = [short_product(p) for p in agreement.index]
    agreement.loc["Macro average"] = agreement.mean().round(3)
    agreement.loc["Random baseline"] = round(1 / (len(agreement) - 1), 3)
    agreement.to_csv(TABLES / "neighbour_agreement.csv", index_label="product")

    props["neighbour_agreement_macro"] = [agreement.loc["Macro average", "tfidf"],
                                          agreement.loc["Macro average", "word2vec"]]
    props.to_csv(TABLES / "vectorizer_comparison.csv", index=False)
    log.info("neighbour agreement (macro): TF-IDF %.3f | Word2Vec %.3f | random %.3f",
             agreement.loc["Macro average", "tfidf"], agreement.loc["Macro average", "word2vec"],
             agreement.loc["Random baseline", "tfidf"])

    # ---- 2-D projections of the same subset, for the figure
    svd = TruncatedSVD(n_components=2, random_state=cfg["seed"]).fit_transform(tfidf[eval_idx])
    pca = PCA(n_components=2, random_state=cfg["seed"]).fit_transform(normalize(doc_vecs[eval_idx]))
    save({"labels": [short_product(p) for p in labels], "tfidf_2d": svd, "w2v_2d": pca},
         "projection.pkl")
