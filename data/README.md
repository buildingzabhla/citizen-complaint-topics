# Data

## Source

The complaints come from the **Consumer Complaint Database of the US Consumer Financial
Protection Bureau (CFPB)**: complaints about financial products and services, each with the
consumer's own description of the problem (the *narrative*) where the consumer agreed to its
publication. The CFPB removes personal information from the narratives before publishing them
and replaces it with `XXXX`; amounts appear as `{$123.00}`.

The texts stand in for the municipal complaints of the use case. They share the properties
that matter for the analysis: unstructured, written informally by members of the public,
very different in length and quality, and skewed by a small number of heavy users.

## Why an archived copy is used

On 5 October 2026 the CFPB's [bulk download](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)
and [API](https://cfpb.github.io/api/ccdb/) no longer contained the narrative column (checked
while building this project: the CSV has 15 columns and no `Consumer complaint narrative`,
and API records have no `complaint_what_happened` field). The narratives therefore come from an
archived copy of the official database:

| | |
|---|---|
| Dataset | [`davidheineman/consumer-finance-complaints-large`](https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large) on Hugging Face |
| Snapshot | complaints received 1 Dec 2011 to 2 Jan 2025 (7,179,332 rows) |
| Pinned revision | `44cfa170a402e254407470275ce05d7dcaccde30` |
| Files | 11 Parquet files, 1.4 GB; SHA-256 of each in `snapshot_manifest.json` |

`python run.py --steps data` downloads exactly this revision and refuses to continue if any
file differs from the manifest.

## Verification

Because the text can no longer be checked against an official source, the pipeline checks
the archived records against the CFPB's **current** official file, which still lists every
complaint with its metadata. Of the 60,000 sampled complaints, 99.7% are still listed there;
for those, date received, product and issue match in 100% of cases and the company in 99.0%
(`results/tables/data_overview.json`). The archived copy is therefore very likely a faithful
copy of the official database.

## Sample

* Window: complaints **received in 2024**, the most recent twelve complete months with
  narratives (`config.yaml: data.date_min / date_max`).
* 680,980 complaints in the window have a narrative. A uniform random sample of **60,000** is
  drawn: each complaint ID gets a stable pseudo-random key (BLAKE2b hash of seed and ID) and the
  60,000 smallest keys are kept. The sample is therefore independent of file order and
  identical on every run with the same seed.
* `sample_complaint_ids.csv` lists the sampled IDs, whether each is a copy of another
  narrative, and its representative, so the exact corpus can be rebuilt.

## Licence

The CFPB publishes the database as a public-domain work of the US government. The archive
used here is published on Hugging Face under Apache-2.0. Neither the raw data nor the
narratives are stored in this repository; they are downloaded by the pipeline.
