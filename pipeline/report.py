"""Step 6 - report: figures, and results/RESULTS.md with every key table.

Topic labels are a human interpretation of the top words. They are kept separately in
results/topic_labels.yaml so that it is always visible what the models produced (top
words) and what was read into them (labels).
"""
from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .utils import FIGURES, RESULTS, TABLES, load, log  # noqa: E402
from .vectorize import short_product  # noqa: E402

NAVY, BLUE, ORANGE, GREY = "#1E2761", "#2A8FB0", "#E07A3F", "#9AA0A6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})
LABELS_FILE = RESULTS / "topic_labels.yaml"


def topic_labels() -> dict:
    if LABELS_FILE.exists():
        return yaml.safe_load(LABELS_FILE.read_text()) or {}
    return {}


def label(labels: dict, model: str, topic: int) -> str:
    return labels.get(model, {}).get(int(topic), f"Topic {int(topic)}")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- figures
def fig_duplicates() -> None:
    t = pd.read_csv(TABLES / "sample_by_product.csv", index_col=0)
    t.index = [short_product(p) for p in t.index]
    t = t.sort_values("sampled")
    unique = t["sampled"] - t["duplicates"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(t.index, unique, color=NAVY, label="unique narratives")
    ax.barh(t.index, t["duplicates"], left=unique, color=ORANGE, label="copies of another narrative")
    for i, (u, d) in enumerate(zip(unique, t["duplicates"])):
        if u + d > 1500:
            ax.text(u + d + 300, i, f"{d / (u + d):.0%} copies", va="center", fontsize=9)
    ax.set_xlabel("complaints in the random sample of 60,000")
    ax.set_title("Half of the sampled narratives are copies of another one")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "duplicates_by_product.png")


def fig_coherence(k_lda: int, k_nmf: int) -> None:
    s = pd.read_csv(TABLES / "coherence_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model, color, k in (("NMF", NAVY, k_nmf), ("LDA", ORANGE, k_lda)):
        d = s[s["model"] == model]
        ax.plot(d["k"], d["c_v"], marker="o", ms=4, color=color, label=model)
        best = d[d["k"] == k]
        ax.scatter(best["k"], best["c_v"], s=140, facecolors="none", edgecolors=color, linewidths=2)
    ax.set_xlabel("number of topics k")
    ax.set_ylabel("C_v coherence")
    ax.set_title("Topic coherence across the number of topics (circle = chosen k)")
    ax.set_xticks(sorted(s["k"].unique()))
    ax.legend(frameon=False)
    lda = s[s["model"] == "LDA"]
    k_best = int(lda.loc[lda["c_v"].idxmax(), "k"])
    if k_best != k_lda:
        fig.text(0.01, -0.04, f"LDA scores highest at k = {k_best}, but those topics include a catch-all; "
                 f"k = {k_lda} was chosen after reading the topics (see config.yaml).",
                 fontsize=8.5, color="#555555")
    _save(fig, "coherence_by_k.png")


def fig_prevalence(model: str, key: str, labels: dict) -> None:
    t = pd.read_csv(TABLES / f"topics_{key}.csv").sort_values("unique_share")
    names = [label(labels, key, k) for k in t["topic"]]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(t) + 1.4))
    ax.barh(y + 0.2, t["unique_share"], height=0.4, color=NAVY, label="unique narratives")
    ax.barh(y - 0.2, t["share_with_duplicates"], height=0.4, color=ORANGE,
            label="all sampled complaints (copies counted)")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of complaints whose dominant topic this is")
    ax.set_title(f"Most prevalent topics - {model}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    _save(fig, f"prevalence_{key}.png")


def fig_top_words(labels: dict) -> None:
    t = load("topics.pkl")
    vocab = load("bow_tfidf.pkl")["vocab"]
    comps = t["nmf_components"]
    order = pd.read_csv(TABLES / "topics_nmf.csv")["topic"].tolist()
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), order):
        ax.axis("on")
        idx = np.argsort(-comps[k])[:8][::-1]
        ax.barh([vocab[i] for i in idx], comps[k][idx], color=BLUE)
        ax.set_title(textwrap.fill(label(labels, "nmf", k), 30), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks([])
    fig.suptitle("NMF topics: the eight highest-weighted terms (ordered by prevalence)",
                 fontweight="bold")
    _save(fig, "nmf_top_words.png")


def fig_vectorizers() -> None:
    a = pd.read_csv(TABLES / "neighbour_agreement.csv", index_col=0)
    base = a.loc["Random baseline", "tfidf"]
    a = a.drop(index=["Random baseline"])
    y = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(y + 0.2, a["tfidf"], height=0.4, color=NAVY, label="TF-IDF")
    ax.barh(y - 0.2, a["word2vec"], height=0.4, color=ORANGE, label="Word2Vec (mean)")
    ax.axvline(base, color=GREY, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(a.index)
    ax.invert_yaxis()
    ax.text(base + 0.01, -0.75, "random", color=GREY, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the 10 nearest neighbours with the same CFPB product")
    ax.set_title("How well each representation groups related complaints")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "vectorizer_neighbour_agreement.png")

    pr = load("projection.pkl")
    labels = np.array(pr["labels"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for ax, key, title in ((axes[0], "tfidf_2d", "TF-IDF (truncated SVD)"),
                           (axes[1], "w2v_2d", "Word2Vec document vectors (PCA)")):
        for i, p in enumerate(sorted(set(labels))):
            m = labels == p
            ax.scatter(pr[key][m, 0], pr[key][m, 1], s=5, alpha=0.5, color=cmap(i), label=p)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(markerscale=3, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("The same 400 complaints per product in two dimensions", fontweight="bold")
    _save(fig, "vectorizer_projection.png")


def fig_heatmap(labels: dict) -> None:
    ct = pd.read_csv(TABLES / "topic_by_product_nmf.csv", index_col=0)
    order = pd.read_csv(TABLES / "topics_nmf.csv")["topic"].tolist()
    ct = ct.loc[[k for k in order if k in ct.index]]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(ct) + 2))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels([label(labels, "nmf", k) for k in ct.index])
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=40, ha="right")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="share of the topic's complaints")
    ax.set_title("Which CFPB products each NMF topic draws on")
    _save(fig, "nmf_topic_by_product.png")


# --------------------------------------------------------------------------- results page
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def topic_md(key: str, labels: dict) -> str:
    t = pd.read_csv(TABLES / f"topics_{key}.csv")
    view = pd.DataFrame({
        "Rank": range(1, len(t) + 1),
        "Topic": [label(labels, key, k) for k in t["topic"]],
        "Top words": t["top_words"],
        "Unique narratives": t["unique_share"].map(_pct),
        "With copies": t["share_with_duplicates"].map(_pct),
        "C_v": t["c_v"],
    })
    return _md(view)


def write_results(k_lda: int, k_nmf: int, labels: dict) -> None:
    ov = json.loads((TABLES / "data_overview.json").read_text())
    c, v = ov["counts"], ov["verification"]
    pre = pd.read_csv(TABLES / "preprocessing_summary.csv").iloc[0]
    metrics = pd.read_csv(TABLES / "summary_metrics.csv")
    agree = pd.read_csv(TABLES / "agreement_with_labels.csv")
    vec = pd.read_csv(TABLES / "vectorizer_comparison.csv")
    neigh = pd.read_csv(TABLES / "word2vec_neighbours.csv")
    between = pd.read_csv(TABLES / "agreement_between_methods.csv", index_col=0)

    lines = [
        "# Results",
        "",
        "_This page is generated by `python run.py --steps report`. All numbers come from the "
        "tables in `results/tables/`; topic labels are a human reading of the top words "
        "(`results/topic_labels.yaml`)._",
        "",
        "## 1. Data",
        "",
        _md(pd.DataFrame({
            "Stage": ["Complaints in the archived database (to 2 Jan 2025)",
                      f"Received {c['date_min']} to {c['date_max']}",
                      "... with a published narrative", "Random sample (seed 42)",
                      "Unique narratives after removing copies",
                      f"Documents with at least {pre.get('min_doc_tokens', 10)} tokens (analysed)"],
            "Complaints": [f"{c['rows_in_snapshot']:,}", f"{c['rows_in_window']:,}",
                           f"{c['with_narrative']:,}", f"{c['sample']:,}",
                           f"{c['unique_narratives']:,}", f"{int(pre['documents']):,}"],
        })),
        "",
        f"**Verification against the current official CFPB file:** {v['found_in_official_file']:,} "
        f"of {v['sample']:,} sampled complaints ({_pct(v['found_in_official_file'] / v['sample'])}) "
        f"are still listed there. Of those, the date matches for {_pct(v['date_matches'])}, the "
        f"product for {_pct(v['product_matches'])}, the issue for {_pct(v['issue_matches'])} and "
        f"the company for {_pct(v['company_matches'])}.",
        "",
        f"**Copies:** {c['duplicates_removed']:,} of {c['sample']:,} sampled narratives "
        f"({_pct(c['duplicates_removed'] / c['sample'])}) are exact or near-identical copies of "
        "another sampled narrative, almost all of them credit-reporting and debt-collection "
        "templates.",
        "",
        "![Duplicates by product](figures/duplicates_by_product.png)",
        "",
        "## 2. Vectorisation: TF-IDF compared with Word2Vec",
        "",
        _md(vec),
        "",
        "Nearest words in the Word2Vec space:",
        "",
        _md(neigh),
        "",
        "![Neighbour agreement](figures/vectorizer_neighbour_agreement.png)",
        "",
        "![Projections](figures/vectorizer_projection.png)",
        "",
        "## 3. Choosing the number of topics",
        "",
        f"Chosen: **LDA k = {k_lda}**, **NMF k = {k_nmf}**.",
        "",
        "![Coherence by k](figures/coherence_by_k.png)",
        "",
        "## 4. The most prevalent topics",
        "",
        f"### NMF (k = {k_nmf})",
        "",
        topic_md("nmf", labels),
        "",
        "![NMF prevalence](figures/prevalence_nmf.png)",
        "",
        "![NMF top words](figures/nmf_top_words.png)",
        "",
        "![NMF topics by product](figures/nmf_topic_by_product.png)",
        "",
        f"### LDA (k = {k_lda})",
        "",
        topic_md("lda", labels),
        "",
        "![LDA prevalence](figures/prevalence_lda.png)",
        "",
        "An interactive view of the LDA model: [`lda_visualisation.html`](lda_visualisation.html) "
        "(download and open in a browser).",
        "",
        f"### k-means on the Word2Vec document vectors (k = {k_nmf})",
        "",
        topic_md("kmeans_word2vec", labels),
        "",
        "## 5. Quality and agreement",
        "",
        _md(metrics),
        "",
        "Agreement of each method's topic assignment with the CFPB labels (never seen by the models):",
        "",
        _md(agree),
        "",
        "Agreement between the methods (NMI):",
        "",
        _md(between.reset_index().rename(columns={"index": "method"})),
        "",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict) -> None:
    t = load("topics.pkl")
    labels = topic_labels()
    fig_duplicates()
    fig_coherence(t["k_lda"], t["k_nmf"])
    fig_prevalence("NMF", "nmf", labels)
    fig_prevalence("LDA", "lda", labels)
    fig_top_words(labels)
    fig_vectorizers()
    fig_heatmap(labels)
    write_results(t["k_lda"], t["k_nmf"], labels)
    log.info("wrote figures and results/RESULTS.md")
