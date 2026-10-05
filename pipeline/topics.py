"""Step 4 - topics: extract the most prevalent topics with LDA and NMF.

* LDA (latent Dirichlet allocation, gensim) treats each document as a mixture of topics
  and each topic as a probability distribution over words. It is fitted to word counts.
* NMF (non-negative matrix factorisation, scikit-learn) factorises the TF-IDF matrix into
  document-topic weights W and topic-term weights H. It makes no probabilistic assumptions
  and tends to give sharper topics on short texts.

For both models the number of topics k is swept from 5 to 20 and each setting is scored
with C_v coherence (how often a topic's top words occur together in the texts). The best k
is then checked by reading the topics: config.yaml allows overriding the automatic choice.

As a third, embedding-based view, k-means groups the Word2Vec document vectors; each
cluster is described by the terms whose TF-IDF weight is most raised inside it.

Prevalence is reported twice: among unique narratives (duplicates removed) and among all
sampled complaints (duplicates counted, each with the topic of its representative). The
gap between the two shows which topics are inflated by bulk-filed templates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel, LdaModel, LdaMulticore
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from .utils import INTERIM, TABLES, exists, load, log, save, timed


# --------------------------------------------------------------------------- models
def fit_lda(bow, dictionary, k: int, seed: int, passes: int, cfg: dict,
            multicore: bool = False):
    """Multicore LDA is fast but varies slightly between runs (two runs with the same seed
    shared 98.6% of their top words), so it is only used for the coherence sweep. The final
    model and the stability refits use single-process LDA, which is fully reproducible."""
    lda_cfg = cfg["topics"]["lda"]
    common = dict(corpus=bow, id2word=dictionary, num_topics=k, passes=passes,
                  iterations=lda_cfg["iterations"], chunksize=2000, random_state=seed,
                  eval_every=None)
    if multicore:
        return LdaMulticore(workers=lda_cfg["workers"], **common)
    return LdaModel(**common)


def lda_words(model, n: int) -> list:
    return [[w for w, _ in model.show_topic(t, topn=n)] for t in range(model.num_topics)]


def lda_doc_topics(model, bow) -> np.ndarray:
    out = np.zeros((len(bow), model.num_topics), dtype=np.float32)
    for i, doc in enumerate(bow):
        for t, p in model.get_document_topics(doc, minimum_probability=0.0):
            out[i, t] = p
    return out


def fit_nmf(tfidf, k: int, seed: int, cfg: dict) -> tuple:
    model = NMF(n_components=k, init="nndsvda", random_state=seed,
                max_iter=cfg["topics"]["nmf"]["max_iter"])
    return model, model.fit_transform(tfidf)


def nmf_words(model, vocab: list, n: int) -> list:
    return [[vocab[i] for i in np.argsort(-row)[:n]] for row in model.components_]


# --------------------------------------------------------------------------- quality
def coherence(topics: list, texts: list, dictionary) -> tuple:
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v")
    per_topic = cm.get_coherence_per_topic()
    return float(np.mean(per_topic)), per_topic


def diversity(topics: list) -> float:
    """Share of distinct words among all topics' top words (1 = no overlap)."""
    words = [w for t in topics for w in t]
    return len(set(words)) / len(words)


def matched_jaccard(a: list, b: list) -> float:
    """Pair the topics of two runs one-to-one and average their top-word overlap."""
    jac = np.array([[len(set(x) & set(y)) / len(set(x) | set(y)) for y in b] for x in a])
    r, c = linear_sum_assignment(-jac)
    return float(jac[r, c].mean())


def choose_k(sweep: pd.DataFrame, model: str, setting) -> int:
    if setting != "auto":
        return int(setting)
    s = sweep[sweep["model"] == model]
    return int(s.loc[s["c_v"].idxmax(), "k"])


# --------------------------------------------------------------------------- prevalence
def prevalence(doc_topics: np.ndarray, corpus: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    dominant = doc_topics.argmax(axis=1)
    k = doc_topics.shape[1]
    unique_counts = np.bincount(dominant, minlength=k)
    # every sampled complaint inherits the topic of its representative narrative
    topic_of = dict(zip(corpus["complaint_id"], dominant))
    raw = sample["representative_id"].map(topic_of).dropna().astype(int)
    raw_counts = np.bincount(raw, minlength=k)
    return pd.DataFrame({
        "topic": range(k),
        "unique_documents": unique_counts,
        "unique_share": (unique_counts / unique_counts.sum()).round(4),
        "share_with_duplicates": (raw_counts / raw_counts.sum()).round(4),
        "mean_weight": (normalize(doc_topics, norm="l1").mean(axis=0)).round(4),
    })


def topic_table(words: list, per_topic_cv: list, prev: pd.DataFrame) -> pd.DataFrame:
    t = prev.copy()
    t["top_words"] = [", ".join(w) for w in words]
    t["c_v"] = np.round(per_topic_cv, 3)
    return t.sort_values("unique_share", ascending=False)


# --------------------------------------------------------------------------- step
def run(cfg: dict) -> None:
    seed, tc = cfg["seed"], cfg["topics"]
    corpus, sample = load("corpus.pkl"), load("sample.pkl")
    bt = load("bow_tfidf.pkl")
    dictionary, vocab, bow, tfidf = bt["dictionary"], bt["vocab"], bt["bow"], bt["tfidf"]
    texts = corpus["tokens"].tolist()
    n_words = tc["top_n_words"]

    rng = np.random.default_rng(seed)
    ref_n = min(tc["coherence_reference_docs"], len(texts))
    reference = [texts[i] for i in sorted(rng.choice(len(texts), ref_n, replace=False))]

    # ---- sweep k for both models (reused if nothing that affects it has changed)
    signature = {"seed": seed, "k_min": tc["k_min"], "k_max": tc["k_max"], "n_words": n_words,
                 "lda": tc["lda"]["passes_sweep"], "iterations": tc["lda"]["iterations"],
                 "nmf": tc["nmf"]["max_iter"], "reference": ref_n, "docs": len(texts),
                 "vocab": len(vocab)}
    cached = load("sweep.pkl") if exists("sweep.pkl") else None
    if cached and cached["signature"] == signature:
        sweep = cached["sweep"]
        log.info("reusing the coherence sweep from data/interim/sweep.pkl")
    else:
        rows = []
        for k in range(tc["k_min"], tc["k_max"] + 1):
            with timed(f"k = {k}"):
                nmf, _ = fit_nmf(tfidf, k, seed, cfg)
                w = nmf_words(nmf, vocab, n_words)
                rows.append({"model": "NMF", "k": k, "c_v": coherence(w, reference, dictionary)[0],
                             "diversity": diversity(nmf_words(nmf, vocab, 25))})
                lda = fit_lda(bow, dictionary, k, seed, tc["lda"]["passes_sweep"], cfg,
                              multicore=True)
                w = lda_words(lda, n_words)
                rows.append({"model": "LDA", "k": k, "c_v": coherence(w, reference, dictionary)[0],
                             "diversity": diversity(lda_words(lda, 25))})
            log.info("k=%d  NMF C_v %.3f | LDA C_v %.3f", k, rows[-2]["c_v"], rows[-1]["c_v"])
        sweep = pd.DataFrame(rows).round(4)
        save({"signature": signature, "sweep": sweep}, "sweep.pkl")
    sweep.to_csv(TABLES / "coherence_sweep.csv", index=False)

    k_lda, k_nmf = choose_k(sweep, "LDA", tc["k_lda"]), choose_k(sweep, "NMF", tc["k_nmf"])
    log.info("chosen number of topics: LDA k=%d, NMF k=%d", k_lda, k_nmf)

    # ---- final models
    with timed(f"final LDA (k={k_lda}, {tc['lda']['passes_final']} passes)"):
        lda = fit_lda(bow, dictionary, k_lda, seed, tc["lda"]["passes_final"], cfg)
        lda_dt = lda_doc_topics(lda, bow)
        lda_w = lda_words(lda, n_words)
        _, lda_cv = coherence(lda_w, texts, dictionary)
    with timed(f"final NMF (k={k_nmf})"):
        nmf, nmf_dt = fit_nmf(tfidf, k_nmf, seed, cfg)
        nmf_w = nmf_words(nmf, vocab, n_words)
        _, nmf_cv = coherence(nmf_w, texts, dictionary)

    # ---- embedding view: k-means on the Word2Vec document vectors
    with timed("k-means on Word2Vec document vectors"):
        vecs = normalize(load("word2vec.pkl")["doc_vectors"])
        km = KMeans(n_clusters=k_nmf, n_init=10, random_state=seed).fit(vecs)
        overall = np.asarray(tfidf.mean(axis=0)).ravel()
        km_w = []
        for c in range(k_nmf):
            inside = np.asarray(tfidf[km.labels_ == c].mean(axis=0)).ravel()
            km_w.append([vocab[i] for i in np.argsort(-(inside - overall))[:n_words]])
        km_dt = np.eye(k_nmf, dtype=np.float32)[km.labels_]
        _, km_cv = coherence(km_w, texts, dictionary)

    # ---- stability: refit with other seeds and compare the topics
    stab = []
    with timed("stability check with further seeds"):
        for s in tc["stability_seeds"]:
            other = fit_lda(bow, dictionary, k_lda, s, tc["lda"]["passes_final"], cfg)
            stab.append({"model": "LDA", "seed": s, "matched_jaccard": matched_jaccard(lda_w, lda_words(other, n_words))})
            other_nmf, _ = fit_nmf(tfidf, k_nmf, s, cfg)
            stab.append({"model": "NMF", "seed": s, "matched_jaccard": matched_jaccard(nmf_w, nmf_words(other_nmf, vocab, n_words))})
    pd.DataFrame(stab).round(3).to_csv(TABLES / "stability.csv", index=False)

    tables = {
        "lda": topic_table(lda_w, lda_cv, prevalence(lda_dt, corpus, sample)),
        "nmf": topic_table(nmf_w, nmf_cv, prevalence(nmf_dt, corpus, sample)),
        "kmeans_word2vec": topic_table(km_w, km_cv, prevalence(km_dt, corpus, sample)),
    }
    for name, t in tables.items():
        t.to_csv(TABLES / f"topics_{name}.csv", index=False)

    lda.save(str(INTERIM / "lda.model"))
    save({"k_lda": k_lda, "k_nmf": k_nmf, "lda_doc_topics": lda_dt, "nmf_doc_topics": nmf_dt,
          "kmeans_labels": km.labels_, "lda_words": lda_w, "nmf_words": nmf_w, "kmeans_words": km_w,
          "nmf_components": nmf.components_}, "topics.pkl")
