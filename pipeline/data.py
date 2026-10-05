"""Step 1 - data: obtain the complaints, draw the sample, verify it, remove duplicates.

Why the data come from an archived copy
---------------------------------------
The concept planned to use the CFPB Consumer Complaint Database. When this step was
implemented (5 October 2026) the CFPB no longer published the complaint narratives:
neither the official bulk download nor the API contained the text column any more.
The narratives therefore come from an archived copy of the official database that was
taken on 2 January 2025 and is pinned to an exact revision (see config.yaml). The
current official download, which still contains the metadata of every complaint, is
used to check that the archived copy is genuine (``verify_against_official``).

Why duplicates are removed
--------------------------
Complaint systems are skewed by heavy users. In the CFPB data many credit-reporting
complaints are near-identical templates, often filed in bulk on behalf of consumers.
Counting each copy would make the templates look like the most important concern, so
exact and near-identical narratives are collapsed to one representative.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import ROOT, TABLES, log, path, save, timed

KEEP_COLUMNS = ["complaint_id", "date_received", "product", "sub_product", "issue",
                "sub_issue", "company", "state", "submitted_via", "complaint_what_happened"]
MANIFEST = ROOT / "data" / "snapshot_manifest.json"
SAMPLE_IDS = ROOT / "data" / "sample_complaint_ids.csv"


# --------------------------------------------------------------------------- download
def _download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    tmp.rename(dest)


def _sha256(file) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_files(cfg: dict) -> list:
    snap = cfg["data"]["snapshot"]
    n = snap["files"]
    return [path(snap["dir"]) / f"train-{i:05d}-of-{n:05d}.parquet" for i in range(n)]


def download(cfg: dict) -> None:
    """Fetch the pinned archived snapshot and the current official file, if missing."""
    snap = cfg["data"]["snapshot"]
    for f in snapshot_files(cfg):
        if not f.exists():
            url = (f"https://huggingface.co/datasets/{snap['hf_dataset']}/resolve/"
                   f"{snap['hf_revision']}/data/{f.name}")
            _download(url, f)
    official = path(cfg["data"]["official"]["zip"])
    if not official.exists():
        _download(cfg["data"]["official"]["url"], official)
    check_manifest(cfg)


def check_manifest(cfg: dict) -> None:
    """Record (first run) or verify (later runs) the SHA-256 of every snapshot file."""
    hashes = {f.name: _sha256(f) for f in snapshot_files(cfg)}
    if MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text())["sha256"]
        bad = [n for n, h in hashes.items() if expected.get(n) != h]
        if bad:
            raise RuntimeError(f"snapshot files differ from the manifest: {bad}")
        log.info("all %d snapshot files match the manifest", len(hashes))
    else:
        snap = cfg["data"]["snapshot"]
        MANIFEST.write_text(json.dumps({"dataset": snap["hf_dataset"],
                                        "revision": snap["hf_revision"],
                                        "sha256": hashes}, indent=2))
        log.info("wrote %s", MANIFEST.relative_to(ROOT))


# --------------------------------------------------------------------------- sampling
def sample_key(ids, seed: int) -> np.ndarray:
    """A stable pseudo-random number for each complaint ID.

    Keeping the complaints with the smallest keys is a uniform random sample that does not
    depend on the order of the files, so the same seed always yields the same sample.
    """
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=8).digest(), "big")
         for i in ids), dtype=np.uint64, count=len(ids))


def draw_sample(cfg: dict, seed: int) -> tuple:
    dmin, dmax = cfg["data"]["date_min"], cfg["data"]["date_max"]
    n = cfg["data"]["sample_size"]
    dataset = ds.dataset([str(f) for f in snapshot_files(cfg)], format="parquet")

    # Pass 1: find eligible complaints (in the window, with a narrative) and keep the n
    # smallest keys. Only IDs and keys are held in memory, never the full text.
    counts = {"rows_in_snapshot": 0, "rows_in_window": 0, "with_narrative": 0}
    best_ids = np.array([], dtype=object)
    best_keys = np.array([], dtype=np.uint64)
    for batch in dataset.to_batches(columns=["complaint_id", "date_received",
                                             "complaint_what_happened"], batch_size=250_000):
        counts["rows_in_snapshot"] += batch.num_rows
        date = pc.utf8_slice_codeunits(batch.column("date_received"), 0, 10)
        in_window = pc.and_(pc.greater_equal(date, dmin), pc.less_equal(date, dmax))
        counts["rows_in_window"] += pc.sum(in_window).as_py() or 0
        has_text = pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(
            pc.fill_null(batch.column("complaint_what_happened"), ""))), 0)
        mask = pc.and_(in_window, has_text)
        ids = pc.filter(batch.column("complaint_id"), mask).to_numpy(zero_copy_only=False)
        counts["with_narrative"] += len(ids)
        if len(ids) == 0:
            continue
        all_ids = np.concatenate([best_ids, ids])
        all_keys = np.concatenate([best_keys, sample_key(ids, seed)])
        if len(all_keys) > n:
            keep = np.argpartition(all_keys, n - 1)[:n]
            all_ids, all_keys = all_ids[keep], all_keys[keep]
        best_ids, best_keys = all_ids, all_keys

    # Pass 2: read the full records of the selected complaints.
    table = dataset.to_table(columns=KEEP_COLUMNS,
                             filter=pc.field("complaint_id").isin(pa.array(best_ids.tolist())))
    df = table.to_pandas().rename(columns={"complaint_what_happened": "narrative"})
    key_of = dict(zip(best_ids, best_keys))
    df["sample_key"] = df["complaint_id"].map(key_of).astype(np.uint64)
    df = df.sort_values("sample_key").reset_index(drop=True)
    counts["sample"] = len(df)
    return df, counts


# --------------------------------------------------------------------------- verification
def verify_against_official(cfg: dict, sample: pd.DataFrame) -> dict:
    """Compare the archived records with the CFPB's current official file.

    The narratives can no longer be checked against an official source, but every complaint
    is still listed there with its date, product, issue and company. If those agree, the
    archived copy is very likely a faithful copy of the official database.
    """
    ids = set(sample["complaint_id"])
    cols = ["Complaint ID", "Date received", "Product", "Issue", "Company"]
    parts = []
    off = cfg["data"]["official"]
    with zipfile.ZipFile(path(off["zip"])) as z, z.open(z.namelist()[0]) as f:
        for chunk in pd.read_csv(f, usecols=cols, dtype=str, na_filter=False,
                                 chunksize=off["chunksize"]):
            hit = chunk["Complaint ID"].isin(ids)
            if hit.any():
                parts.append(chunk[hit])
    official = pd.concat(parts) if parts else pd.DataFrame(columns=cols)
    m = sample.merge(official, left_on="complaint_id", right_on="Complaint ID", how="left")
    found = m["Complaint ID"].notna()
    f = m[found]
    result = {
        "sample": len(sample),
        "found_in_official_file": int(found.sum()),
        "date_matches": float((f["date_received"].str[:10] == f["Date received"]).mean()),
        "issue_matches": float((f["issue"] == f["Issue"]).mean()),
        "company_matches": float((f["company"] == f["Company"]).mean()),
        "product_matches": float((f["product"] == f["Product"]).mean()),
    }
    # Product categories can be renamed over time; record what differs.
    renamed = (f[f["product"] != f["Product"]].groupby(["product", "Product"]).size()
               .sort_values(ascending=False).head(10))
    result["product_differences"] = [
        {"archived": a, "official_now": b, "count": int(c)} for (a, b), c in renamed.items()]
    return result


# --------------------------------------------------------------------------- duplicates
_AMOUNT = re.compile(r"\{\$?[\d,.]*\}")
_REDACTION = re.compile(r"x{2,}")
_NOT_LETTER = re.compile(r"[^a-z]+")


def normalise_for_dedup(text: str) -> str:
    """Lowercase, then drop amounts, redactions (XXXX), digits and punctuation."""
    t = _AMOUNT.sub(" ", text.lower())
    t = _REDACTION.sub(" ", t)
    return " ".join(_NOT_LETTER.sub(" ", t).split())


def find_duplicates(texts: list, order: np.ndarray, threshold: float,
                    chunk: int = 500) -> np.ndarray:
    """Group exact and near-identical texts; return each text's representative index.

    ``order`` ranks the texts (smaller = preferred); the representative of a group is the
    member with the smallest rank, which keeps the result deterministic.
    """
    norm = [normalise_for_dedup(t) for t in texts]
    # exact duplicates: identical normalised text
    first_of = {}
    exact_rep = np.empty(len(norm), dtype=np.int64)
    for i in np.argsort(order, kind="stable"):
        exact_rep[i] = first_of.setdefault(norm[i], i)
    uniq = np.unique(exact_rep)

    # near duplicates among the remaining distinct texts: cosine similarity of TF-IDF vectors
    vec = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    try:
        X = vec.fit_transform([norm[i] for i in uniq])
    except ValueError:      # no shared vocabulary left (only happens on tiny inputs)
        return exact_rep
    XT = X.T.tocsc()
    rows, cols = [], []
    for start in range(0, X.shape[0], chunk):
        sim = (X[start:start + chunk] @ XT).tocoo()
        r = sim.row + start
        keep = (sim.data >= threshold) & (r < sim.col)
        rows.append(r[keep])
        cols.append(sim.col[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uniq), len(uniq)))
    _, component = connected_components(graph, directed=False)

    # representative of each component = member with the smallest rank
    best = {}
    for j, c in enumerate(component):
        i = uniq[j]
        if c not in best or order[i] < order[best[c]]:
            best[c] = i
    near_rep = {uniq[j]: best[c] for j, c in enumerate(component)}
    return np.array([near_rep[exact_rep[i]] for i in range(len(norm))])


# --------------------------------------------------------------------------- step
def run(cfg: dict) -> None:
    seed = cfg["seed"]
    with timed("download / verify files"):
        download(cfg)
    with timed("draw the random sample"):
        sample, counts = draw_sample(cfg, seed)
    log.info("snapshot rows %(rows_in_snapshot)d | in window %(rows_in_window)d | "
             "with narrative %(with_narrative)d | sampled %(sample)d", counts)

    with timed("verify against the official CFPB file"):
        verification = verify_against_official(cfg, sample)
    log.info("found %d of %d sampled IDs in the official file; dates match %.1f%%, "
             "issues %.1f%%, companies %.1f%%", verification["found_in_official_file"],
             verification["sample"], 100 * verification["date_matches"],
             100 * verification["issue_matches"], 100 * verification["company_matches"])

    with timed("find exact and near-duplicate narratives"):
        rep = find_duplicates(sample["narrative"].tolist(), sample["sample_key"].to_numpy(),
                              cfg["data"]["near_duplicate_threshold"])
    sample["representative_id"] = sample["complaint_id"].to_numpy()[rep]
    sample["is_duplicate"] = sample["representative_id"] != sample["complaint_id"]
    sample["group_size"] = sample.groupby("representative_id")["complaint_id"].transform("size")
    n_dup = int(sample["is_duplicate"].sum())
    log.info("%d of %d sampled narratives (%.1f%%) duplicate another narrative",
             n_dup, len(sample), 100 * n_dup / len(sample))

    save(sample, "sample.pkl")
    sample[["complaint_id", "is_duplicate", "representative_id"]].to_csv(SAMPLE_IDS, index=False)

    overview = dict(counts)
    overview.update({"duplicates_removed": n_dup, "unique_narratives": len(sample) - n_dup,
                     "date_min": cfg["data"]["date_min"], "date_max": cfg["data"]["date_max"],
                     "seed": seed})
    (TABLES / "data_overview.json").write_text(json.dumps(
        {"counts": overview, "verification": verification}, indent=2))

    by_product = (sample.groupby("product")
                  .agg(sampled=("complaint_id", "size"), duplicates=("is_duplicate", "sum"))
                  .sort_values("sampled", ascending=False))
    by_product["duplicate_share"] = (by_product["duplicates"] / by_product["sampled"]).round(3)
    by_product["share_of_sample"] = (by_product["sampled"] / len(sample)).round(3)
    by_product.to_csv(TABLES / "sample_by_product.csv")
