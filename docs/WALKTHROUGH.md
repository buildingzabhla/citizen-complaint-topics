# Walkthrough: what each step does and why

This document explains the pipeline step by step, with the reasoning behind every decision,
so that the project can be followed, checked and extended. The code lives in `pipeline/`;
every parameter mentioned here is set in `config.yaml`.

---

## The question

A municipality collects written complaints online, but there are too many to read and they
are unstructured. Decision-makers want to know **which topics citizens raise most often**.
This project builds and tests the analysis on consumer complaints to the US CFPB, which share
the relevant properties: free text from the public, very uneven in length and quality, and
skewed by a few heavy users.

The task is unsupervised. Nobody has labelled the complaints by topic, and the topics are
not known in advance, so they have to be discovered from the text itself.

---

## Step 1: data (`pipeline/data.py`)

**Getting the text.** The concept planned to use the CFPB database. When the step was
implemented, the CFPB no longer published the complaint narratives: the official download and
the API still list every complaint, but without the text. The narratives therefore come from
an archived copy of the official database, taken on 2 January 2025 and pinned to an exact
revision, with a SHA-256 checksum for every file (`data/snapshot_manifest.json`).

**Checking the archived copy.** A copy made by a third party could in principle be altered.
The official file still contains the metadata of every complaint, so the pipeline compares
the two: 99.7% of the sampled complaints are still listed officially, and for those the date,
product and issue agree in 100% of cases. That is strong evidence that the copy is genuine.

**Time window.** The use case is about *currently* discussed topics, so the analysis uses one
recent year: complaints received in 2024, the most recent full year with narratives.

**Sampling.** 680,980 complaints from 2024 have a narrative. Processing all of them would be
slow without changing the answer, so 60,000 are drawn at random. Each complaint ID gets a
stable pseudo-random key (a hash of the seed and the ID), and the 60,000 smallest keys are
kept. This is a uniform random sample that comes out identical on every run and does not
depend on the order of the files.

**Duplicates, the most important data-quality finding.** 50.8% of the sampled narratives
are exact or near-identical copies of another sampled narrative. The largest single template
appears 3,004 times in the sample. Almost all copies are credit-reporting (61% of that
product) and debt-collection (34%) complaints: letter templates, typically filed in bulk on
behalf of consumers. This is the problem the brief describes, where a few heavy users make
the data unrepresentative. Counting every copy would make the templates look like the
public's main concern, so each group of copies is reduced to one representative narrative.

*How copies are found:* texts are normalised (lowercase; redactions, amounts, digits and
punctuation removed). Identical normalised texts are exact copies. The remaining texts are
compared by the cosine similarity of their TF-IDF vectors, and pairs above 0.9 are linked.
Linked texts form groups (connected components), and each group keeps the member with the
smallest sampling key. Because groups are transitive, two variants of one template that differ
more (for example a long and a short list of accounts) still end up in the same group through
the variants in between.

---

## Step 2: preprocessing (`pipeline/preprocess.py`)

Worked example (an invented complaint, run through the real code):

| | |
|---|---|
| Raw | On XX/XX/2024 I noticed a hard inquiry from XXXX on my credit report that I never authorized. I disputed it twice with the credit bureaus, but they said the account is verified and refused to remove the late payments of {$450.00}. |
| 1. Cleaned | On I noticed a hard inquiry from on my credit report that I never authorized I disputed it twice with the credit bureaus but they said the account is verified and refused to remove the late payments of |
| 2–4. Tokens | notice, hard, inquiry, credit, report, authorize, dispute, credit, bureaus, account, verify, refuse, remove, late, payment |

1. **Clean (regular expressions).** Remove what the CFPB inserts for privacy (`XXXX`,
   `{$amounts}`), links, e-mail addresses, digits and punctuation.
2. **Tokenise, tag, lemmatise (spaCy).** Lemmatising maps inflected forms to one base form
   ("disputed" to "dispute", "payments" to "payment"), so they count as one word. Lemmas are
   lowercased *after* tagging, because spaCy uses capital letters to recognise names.
3. **Keep content words.** Nouns, proper nouns, verbs and adjectives carry the topics;
   pronouns, adverbs, numbers and similar words mostly add noise.
4. **Remove stop words.** NLTK's English list plus a short domain list of words that appear
   in almost every complaint without saying anything about its topic (e.g. *tell, say, call,
   would, please*). They are removed after lemmatisation, so every inflected form is caught.
5. **Join frequent two-word expressions (gensim Phrases).** For example `late_payment`,
   `identity_theft`, `hard_inquiry`, `debt_collector`. This gives the vocabulary meaningful
   bigrams alongside single words.
6. **Drop very short documents.** The concept planned a minimum of 20 tokens. During
   development this turned out to remove 27% of the unique narratives, and unevenly: 30% of
   credit-reporting and debt-collection complaints but only 9% of mortgage complaints. Most
   of the removed texts had 10–19 content words, enough to carry a topic. Keeping the
   threshold would have distorted topic prevalence, so it was lowered to **10 tokens**.
   Result: 26,727 documents are analysed.

---

## Step 3: vectorisation (`pipeline/vectorize.py`)

Topic models need numbers, not words. Two representations are built and compared.

**Bag of words and TF-IDF.** Each document is a vector with one entry per vocabulary term
(5,653 terms that occur in at least 10 documents and in at most half of them). LDA uses the
raw *counts*, because it is a probabilistic model of how often words occur. NMF uses
**TF-IDF** weights, which give less weight to words that appear in many documents. The
vectors are sparse: only 0.7% of the entries are non-zero.

**Word2Vec.** A skip-gram model learns a 100-dimensional vector for every word from the words
around it, so words used in similar contexts get similar vectors. A document is represented
by the average of its word vectors. The model learned, for example, that *refund* is close to
*reimburse* and *chargeback*, and it even placed misspellings next to the right words
(*mortage* next to *mortgage*, *idenity* next to *identity_theft*), which a bag of words
treats as unrelated terms.

**How they compare.** For a balanced subset of 400 complaints per product, each complaint's
10 nearest neighbours are found in each representation, and we measure how many share its
CFPB product. The product label is never used to build the vectors, so this shows how much
topical information each one captures. Word2Vec scores 0.62 and TF-IDF 0.57, against 0.125
by chance. Word2Vec groups related complaints slightly better, because it recognises related
words. TF-IDF's advantage is interpretability: every dimension is a word, which is why it
feeds the NMF topic model.

---

## Step 4: topics (`pipeline/topics.py`)

**LDA** (latent Dirichlet allocation) assumes every document is a mixture of topics and every
topic is a probability distribution over words, and infers both from the word counts.

**NMF** (non-negative matrix factorisation) splits the TF-IDF matrix into two smaller
non-negative matrices: how strongly each document belongs to each topic, and how strongly each
word belongs to each topic. It makes no probabilistic assumptions and often gives sharper
topics on short texts.

**Choosing the number of topics k.** Too few topics blur different concerns together; too many
split one concern into fragments. Both models are fitted for every k from 5 to 20 and scored
with **C_v coherence**, which measures how often a topic's top words actually occur together
in the texts. People tend to find topics with high coherence easier to interpret. The
highest-scoring k is then checked by reading the topics.

**A third view: clusters of Word2Vec vectors.** k-means groups the document vectors. Each
cluster is described by the words whose TF-IDF weight rises most inside it. This shows
whether the embedding representation finds a similar structure.

**Prevalence, counted two ways.** A complaint's *dominant topic* is the topic with the
highest weight. Prevalence is the share of complaints per dominant topic, counted:

* among **unique narratives**: this is the answer to the question;
* among **all sampled complaints**, with every copy taking its representative's topic. The
  difference shows which topics bulk-filed templates inflate.

---

## Step 5: evaluation (`pipeline/evaluate.py`)

Unsupervised models have no accuracy score, so several signals are combined:

| Signal | What it shows |
|---|---|
| C_v coherence | whether a topic's top words belong together |
| Diversity | whether topics are distinct (share of unique words among all top words) |
| Stability | whether the same topics come back with other random seeds (matched Jaccard overlap of top words) |
| NMI / ARI with CFPB product and issue | whether topics line up with categories the models never saw |
| NMI between methods | whether LDA, NMF and the embedding clusters find a similar structure |

---

## Decisions that changed during development

| Concept (phase 1) | Implementation (phase 2) | Reason |
|---|---|---|
| CFPB data via Kaggle | archived copy of the official database (2 Jan 2025), verified against the current official file | the CFPB stopped publishing narratives; the Kaggle copies are older |
| — | exact and near-duplicate removal | 50.8% of sampled narratives are copies of others |
| minimum 20 tokens per document | minimum 10 tokens | 20 removed 27% of documents and distorted prevalence between products |
| stop words, then lemmatisation | lemmatisation, then stop words; content words only | catches every inflected form; focuses topics on content words |
| TF-IDF as input for the topic models | LDA on counts, NMF on TF-IDF | LDA is a model of word counts |
| (not specified) | single-process LDA | multicore LDA was not reproducible: in a second run its coherence scores moved by up to 0.018 and the best k changed from 5 to 15 |

---

## Known limitations

* **Proxy domain.** Financial complaints from the US, not municipal complaints. The *method*
  carries over; the topics themselves do not.
* **Who complains.** People who file formal complaints are not a random sample of the public,
  and only narratives the consumer agreed to publish are included.
* **Templates.** Removing copies counts each template once. If many different people genuinely
  used the same template for their own problem, their concern is undercounted. The second
  prevalence column shows the effect.
* **Preprocessing.** spaCy's small model misses some lemmas (*bureaus* and *bureau* remain
  separate). The phrase detector does not join very common pairs (*credit report* stays two
  words).
* **Dominant topic.** Assigning each complaint to one topic simplifies mixed complaints.
  LDA's topic mixtures are kept in `data/interim/` for finer analysis.
