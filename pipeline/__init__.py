"""Topic analysis of consumer complaint narratives.

The package is organised by pipeline step:

    data.py        obtain the complaints, draw the sample, verify it, remove duplicates
    preprocess.py  turn raw narratives into clean, lemmatised tokens
    vectorize.py   bag-of-words / TF-IDF and Word2Vec representations, and their comparison
    topics.py      topic extraction with LDA and NMF (plus clustering of the embeddings)
    evaluate.py    coherence, diversity, stability and agreement with the CFPB labels
    report.py      figures, tables and the results summary

run.py in the repository root executes the steps in order.
"""
