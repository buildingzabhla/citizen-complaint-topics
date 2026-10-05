"""Step 5 - evaluation: how good and how trustworthy are the topics?

Unsupervised topic models have no accuracy score, so several weaker signals are combined:

* C_v coherence per topic and on average (computed in the topics step),
* topic diversity (overlap of the top words between topics),
* stability across random seeds (computed in the topics step),
* agreement with metadata the models never saw: the CFPB product and issue labels
  (normalised mutual information, NMI, and adjusted Rand index, ARI),
* agreement between the three methods, i.e. whether they find a similar structure.

An interactive pyLDAvis view of the LDA model is written to results/ as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models import LdaModel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .utils import INTERIM, RESULTS, TABLES, load, log, timed
from .vectorize import short_product


def run(cfg: dict) -> None:
    corpus, t, bt = load("corpus.pkl"), load("topics.pkl"), load("bow_tfidf.pkl")
    product = corpus["product"].map(short_product).to_numpy()
    issue = corpus["issue"].fillna("").to_numpy()
    assign = {
        "LDA": t["lda_doc_topics"].argmax(axis=1),
        "NMF": t["nmf_doc_topics"].argmax(axis=1),
        "k-means on Word2Vec": np.asarray(t["kmeans_labels"]),
    }

    rows = []
    for name, a in assign.items():
        rows.append({"method": name, "topics": int(len(np.unique(a))),
                     "nmi_product": normalized_mutual_info_score(product, a),
                     "ari_product": adjusted_rand_score(product, a),
                     "nmi_issue": normalized_mutual_info_score(issue, a)})
    pd.DataFrame(rows).round(3).to_csv(TABLES / "agreement_with_labels.csv", index=False)

    names = list(assign)
    between = pd.DataFrame(
        [[normalized_mutual_info_score(assign[a], assign[b]) for b in names] for a in names],
        index=names, columns=names).round(3)
    between.to_csv(TABLES / "agreement_between_methods.csv")

    for model in ("NMF", "LDA"):
        ct = pd.crosstab(assign[model], product, normalize="index").round(3)
        ct.index.name = "topic"
        ct.to_csv(TABLES / f"topic_by_product_{model.lower()}.csv")

    summary = []
    stab = pd.read_csv(TABLES / "stability.csv")
    for model, key in (("LDA", "lda"), ("NMF", "nmf"), ("k-means on Word2Vec", "kmeans_word2vec")):
        topics = pd.read_csv(TABLES / f"topics_{key}.csv")
        words = [w for row in topics["top_words"] for w in row.split(", ")]
        s = stab.loc[stab["model"] == model, "matched_jaccard"]
        summary.append({"method": model, "topics": len(topics), "mean_c_v": topics["c_v"].mean(),
                        "diversity_top10": len(set(words)) / len(words),
                        "stability_jaccard": s.mean() if len(s) else np.nan})
    pd.DataFrame(summary).round(3).to_csv(TABLES / "summary_metrics.csv", index=False)

    with timed("interactive LDA visualisation (pyLDAvis)"):
        lda = LdaModel.load(str(INTERIM / "lda.model"))
        vis = pyLDAvis.gensim_models.prepare(lda, bt["bow"], bt["dictionary"], sort_topics=False)
        pyLDAvis.save_html(vis, str(RESULTS / "lda_visualisation.html"))
    log.info("wrote evaluation tables and results/lda_visualisation.html")
