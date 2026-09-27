"""Candidate generation (blocking).

For every Source 1 record we retrieve the nearest Source 2 / Source 3 records of
the same country through two independent sparse-vector channels:

  * name channel   : TF-IDF over character 3-grams of ``name_compact``
                     (robust to typos, word-order swaps, glued domain-style names)
  * address channel: TF-IDF over word unigrams of ``addr_norm`` + ``addr_nums``
                     (house number + street + locality tokens)

Both channels run inside (country, state) partitions.  Source 2/3 records whose
state could not be resolved (missing / truncated addresses) are appended to
every state partition of their country so that they can still be reached.
Source 1 records with no resolvable state are compared against the whole country.

The union of the two channels is the candidate set (``candidate_pairs.tsv``).
"""
from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _topn(A: sp.csr_matrix, B: sp.csr_matrix, top_n: int, threshold: float, threads: int):
    """Return (row_idx, col_idx, sim) for the top-n most similar B rows per A row."""
    if A.shape[0] == 0 or B.shape[0] == 0 or A.shape[1] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=top_n, threshold=threshold, sort=True, n_threads=threads)
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def _channel(a_text, b_text, analyzer, ngram, top_n, threshold, threads, min_df=1):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, dtype=np.float32,
                          token_pattern=r"\S+" if analyzer == "word" else None, lowercase=False,
                          sublinear_tf=True)
    try:
        Bm = vec.fit_transform(b_text)
    except ValueError:  # empty vocabulary
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    Am = vec.transform(a_text)
    return _topn(Am.tocsr(), Bm.tocsr(), top_n, threshold, threads)


# States that are routinely confused in the data and therefore share a partition:
# Telangana was carved out of Andhra Pradesh (Hyderabad records carry either label),
# and "Washington" (city) is sometimes resolved to the WA state when the DC code is missing.
PARTITION_MERGE = {"TG": "AP", "DC": "WA"}


def _partition_key(col: pl.Expr) -> pl.Expr:
    return col.replace(PARTITION_MERGE)


def _channel_both(a_text, b_text, analyzer, ngram, top_n, rev_n, threshold, threads):
    """Forward (A->B, top_n per A row) and reverse (B->A, rev_n per B row) retrieval."""
    # max_df: tokens present in more than 10 % of the partition (e.g. "delhi" in Delhi) carry
    # almost no IDF weight but make the sparse product quadratic; drop them.
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, dtype=np.float32,
                          token_pattern=r"\S+" if analyzer == "word" else None, lowercase=False,
                          sublinear_tf=True, max_df=0.1 if len(b_text) > 1000 else 1.0)
    try:
        Bm = vec.fit_transform(b_text).tocsr()
    except ValueError:  # empty vocabulary
        return []
    Am = vec.transform(a_text).tocsr()
    out = [_topn(Am, Bm, top_n, threshold, threads)]
    if rev_n > 0:
        c, r, s = _topn(Bm, Am, rev_n, threshold, threads)  # rows are B, cols are A
        out.append((r, c, s))
    return out


def run_blocking(a: pl.DataFrame, b: pl.DataFrame, top_name: int = 20, top_addr: int = 15,
                 thr_name: float = 0.25, thr_addr: float = 0.2, threads: int = 8,
                 rev_name: int = 3, rev_addr: int = 3, verbose: bool = True) -> pl.DataFrame:
    """a: normalised Source 1 frame, b: normalised Source 2+3 frame.

    Forward channels retrieve ``top_name`` / ``top_addr`` candidates per Source 1
    record; reverse channels retrieve ``rev_name`` / ``rev_addr`` Source 1 records
    per Source 2/3 record so that every candidate record reaches at least a few
    reference entities even inside dense same-name neighbourhoods.

    Returns a frame with columns s1_id, cand_id, sim_name, sim_addr.
    """
    keep = ["country", "state", "name_compact", "addr_norm", "addr_nums"]
    a = a.select(keep).with_row_index("a_idx").with_columns(state=_partition_key(pl.col("state")))
    b = b.select(keep).with_row_index("b_idx").with_columns(state=_partition_key(pl.col("state")))
    b = b.with_columns(addr_text=(pl.col("addr_norm") + " " + pl.col("addr_nums")).str.strip_chars())
    a = a.with_columns(addr_text=(pl.col("addr_norm") + " " + pl.col("addr_nums")).str.strip_chars())
    out_parts = []
    t0 = time.time()
    for country in a["country"].unique().sort():
        a_c = a.filter(pl.col("country") == country)
        b_c = b.filter(pl.col("country") == country)
        b_nostate = b_c.filter(pl.col("state") == "")
        states = a_c["state"].unique().sort().to_list()
        for st in states:
            a_s = a_c.filter(pl.col("state") == st)
            if st == "":
                b_s = b_c
            else:
                b_s = pl.concat([b_c.filter(pl.col("state") == st), b_nostate])
            if a_s.height == 0 or b_s.height == 0:
                continue
            a_idx = a_s["a_idx"].to_numpy()
            b_idx = b_s["b_idx"].to_numpy()
            part_parts = []
            # name channel (forward: per Source 1 row, reverse: per candidate row)
            for r, c, s in _channel_both(a_s["name_compact"].to_list(), b_s["name_compact"].to_list(),
                                         "char", (3, 3), top_name, rev_name, thr_name, threads):
                if len(r):
                    part_parts.append(pl.DataFrame({"a_idx": a_idx[r], "b_idx": b_idx[c], "sim_name": s,
                                                    "sim_addr": np.zeros(len(r), np.float32)}))
            # address channel
            for r, c, s in _channel_both(a_s["addr_text"].to_list(), b_s["addr_text"].to_list(),
                                         "word", (1, 1), top_addr, rev_addr, thr_addr, threads):
                if len(r):
                    part_parts.append(pl.DataFrame({"a_idx": a_idx[r], "b_idx": b_idx[c],
                                                    "sim_name": np.zeros(len(r), np.float32), "sim_addr": s}))
            if part_parts:
                # a Source 1 row lives in exactly one partition, so de-duplicating here is complete
                out_parts.append(pl.concat(part_parts).group_by("a_idx", "b_idx")
                                 .agg(pl.col("sim_name").max(), pl.col("sim_addr").max())
                                 .with_columns(pl.col("a_idx").cast(pl.UInt32), pl.col("b_idx").cast(pl.UInt32)))
            if verbose:
                print(f"  {country:6} {st or '--':4} A={a_s.height:>7,} B={b_s.height:>9,} "
                      f"pairs={sum(p.height for p in out_parts):>11,} t={time.time() - t0:7.1f}s", flush=True)
    if not out_parts:
        return pl.DataFrame({"a_idx": pl.Series([], dtype=pl.UInt32), "b_idx": pl.Series([], dtype=pl.UInt32),
                             "sim_name": pl.Series([], dtype=pl.Float32), "sim_addr": pl.Series([], dtype=pl.Float32)})
    return pl.concat(out_parts)


def gt_pairs_idx(gt: pl.DataFrame, a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth pairs expressed as (a_idx, b_idx) row indices of frames a and b."""
    g = (gt.fill_null("").with_columns(pl.col("matched_entity_ids").str.split(","))
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"}))
    ai = a.select("entity_id", "a_idx") if "a_idx" in a.columns else a.select("entity_id").with_row_index("a_idx")
    bi = b.select("entity_id", "b_idx") if "b_idx" in b.columns else b.select("entity_id").with_row_index("b_idx")
    return (g.join(ai, left_on="s1_id", right_on="entity_id", how="inner")
             .join(bi, left_on="cand_id", right_on="entity_id", how="inner")
             .select("a_idx", "b_idx"))


def evaluate_blocking(pairs: pl.DataFrame, gt: pl.DataFrame, a: pl.DataFrame, b: pl.DataFrame) -> None:
    """Print candidate-set recall for the Source 1 rows of ``a``."""
    g = gt_pairs_idx(gt, a, b)
    hit = g.join(pairs.select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="inner")
    n_true = g.height
    print(f"blocking recall: {hit.height:,}/{n_true:,} = {hit.height / max(n_true, 1):.4f}; "
          f"candidates/S1 = {pairs.height / max(a.height, 1):.1f}; total pairs = {pairs.height:,}")
    for ch in ("sim_name", "sim_addr"):
        h = g.join(pairs.filter(pl.col(ch) > 0).select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="inner")
        print(f"   {ch} channel recall: {h.height / max(n_true, 1):.4f}")
    per = (g.join(pairs.select("a_idx", "b_idx").with_columns(hit=pl.lit(1)), on=["a_idx", "b_idx"], how="left")
             .group_by("a_idx").agg(pl.col("hit").fill_null(0).mean().alias("r")))
    print(f"   S1 entities with full recall: {(per['r'] == 1).mean():.4f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="normalised source1 parquet")
    ap.add_argument("--b", nargs="+", required=True, help="normalised source2/3 parquets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="random sample of Source 1 rows (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gt", default=None, help="ground truth tsv for recall evaluation")
    ap.add_argument("--top-name", type=int, default=20)
    ap.add_argument("--top-addr", type=int, default=15)
    ap.add_argument("--rev-name", type=int, default=3)
    ap.add_argument("--rev-addr", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    a = pl.read_parquet(args.a)
    if args.sample:
        a = a.sample(n=args.sample, seed=args.seed)
    b = pl.concat([pl.read_parquet(p) for p in args.b])
    print(f"A={a.height:,}  B={b.height:,}")
    pairs = run_blocking(a, b, top_name=args.top_name, top_addr=args.top_addr, threads=args.threads,
                         rev_name=args.rev_name, rev_addr=args.rev_addr)
    pairs.write_parquet(args.out)
    print(f"wrote {pairs.height:,} candidate pairs -> {args.out}")
    if args.gt:
        gt = pl.read_csv(args.gt, separator="\t", quote_char=None, infer_schema=False)
        evaluate_blocking(pairs, gt, a, b)
