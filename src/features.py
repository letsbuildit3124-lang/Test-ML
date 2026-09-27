"""Pairwise feature engineering for (Source 1, candidate) pairs.

Three groups of features are produced:

  1. Name similarity   – rapidfuzz ratios on the core / compact / full names,
                         Jaro-Winkler, token Jaccard, first-token agreement,
                         script and domain-style flags, TF-IDF cosine (blocking).
  2. Address similarity– rapidfuzz ratios, token Jaccard, digit-token Jaccard,
                         house-number and state agreement, missingness flags,
                         TF-IDF cosine (blocking).
  3. Context features  – how ambiguous the Source 1 name / address is inside its
                         country (number of reference entities sharing it),
                         per-entity ranks and gaps to the best candidate, and how
                         strongly the candidate is claimed by *other* Source 1
                         entities (one-to-one competition).

All features are language-agnostic string measures so that the model trained on
US + India transfers to unseen countries (France in the test set).

Pairs are keyed by integer row indices (a_idx into the Source 1 frame, b_idx into
the Source 2+3 frame) and processed in partition chunks so that only a bounded
number of string pairs is materialised at any time.
"""
from __future__ import annotations

from multiprocessing import Pool
from typing import Iterator

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

PAIR_COLS = [
    "n_ratio", "n_tsr", "n_tsort", "n_partial", "n_cmp_ratio", "n_jw", "n_jacc", "n_full_ratio",
    "n_first_eq", "n_len_a", "n_len_b", "n_prefix", "n_cmp_eq",
    "a_ratio", "a_tsr", "a_tsort", "a_partial", "a_jacc", "a_num_jacc", "a_num_inter", "a_len_a", "a_len_b",
    "house_rel", "house_ratio", "state_rel",
]
CTX_COLS = ["sim_name", "sim_addr", "amb_name", "amb_addr", "b_nonlatin", "b_domain", "b_addr_missing"]
GROUP_COLS = ["combo", "n_cands", "n_tsr_max", "a_tsr_max", "combo_max", "n_tsr_rank", "a_tsr_rank",
              "combo_rank", "n_hi_name", "n_hi_addr", "n_tsr_gap", "a_tsr_gap", "combo_gap",
              "b_deg", "b_combo_max", "b_combo_rank", "b_combo_gap"]
FEATURE_COLS = CTX_COLS + PAIR_COLS + GROUP_COLS

STR_COLS = ["name_core", "name_compact", "name_norm", "addr_norm", "addr_nums", "house_no", "state"]
A_COLS = STR_COLS + ["country"]
B_COLS = STR_COLS + ["name_script", "name_domain", "addr_missing"]


def _jacc(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pair_feats(row) -> tuple:
    (nc_a, ncp_a, nn_a, ad_a, nums_a, h_a, st_a,
     nc_b, ncp_b, nn_b, ad_b, nums_b, h_b, st_b) = row
    n_ratio = fuzz.ratio(nc_a, nc_b)
    n_tsr = fuzz.token_set_ratio(nc_a, nc_b)
    n_tsort = fuzz.token_sort_ratio(nc_a, nc_b)
    n_partial = fuzz.partial_ratio(nc_a, nc_b)
    n_cmp_ratio = fuzz.ratio(ncp_a, ncp_b)
    n_jw = JaroWinkler.similarity(ncp_a, ncp_b) * 100
    n_jacc = _jacc(nc_a, nc_b)
    n_full_ratio = fuzz.ratio(nn_a, nn_b)
    ta, tb = nc_a.split(), nc_b.split()
    n_first_eq = 1.0 if ta and tb and ta[0] == tb[0] else 0.0
    n_prefix = 1.0 if (ncp_a and ncp_b and (ncp_a.startswith(ncp_b) or ncp_b.startswith(ncp_a))) else 0.0
    n_cmp_eq = 1.0 if ncp_a == ncp_b and ncp_a else 0.0
    if ad_a and ad_b:
        a_ratio = fuzz.ratio(ad_a, ad_b)
        a_tsr = fuzz.token_set_ratio(ad_a, ad_b)
        a_tsort = fuzz.token_sort_ratio(ad_a, ad_b)
        a_partial = fuzz.partial_ratio(ad_a, ad_b)
        a_jacc = _jacc(ad_a, ad_b)
    else:
        a_ratio = a_tsr = a_tsort = a_partial = -1.0
        a_jacc = -1.0
    sa, sb = set(nums_a.split()), set(nums_b.split())
    if sa and sb:
        a_num_inter = float(len(sa & sb))
        a_num_jacc = a_num_inter / len(sa | sb)
    else:
        a_num_inter, a_num_jacc = 0.0, -1.0
    if h_a and h_b:
        house_rel = 1.0 if h_a == h_b else -1.0
        house_ratio = fuzz.ratio(h_a, h_b)
    else:
        house_rel, house_ratio = 0.0, -1.0
    if st_a and st_b:
        state_rel = 1.0 if st_a == st_b else -1.0
    else:
        state_rel = 0.0
    return (n_ratio, n_tsr, n_tsort, n_partial, n_cmp_ratio, n_jw, n_jacc, n_full_ratio,
            n_first_eq, float(len(ta)), float(len(tb)), n_prefix, n_cmp_eq,
            a_ratio, a_tsr, a_tsort, a_partial, a_jacc, a_num_jacc, a_num_inter,
            float(len(ad_a.split())), float(len(ad_b.split())),
            house_rel, house_ratio, state_rel)


def _worker(rows):
    return np.asarray([_pair_feats(r) for r in rows], dtype=np.float32)


def context_frames(a: pl.DataFrame, b: pl.DataFrame, a_full: pl.DataFrame | None = None):
    """Slim frames (with row indices) holding the columns needed for features.

    ``a_full`` (default: ``a``) is the complete reference source used to count how
    many Source 1 entities share a name / address inside a country (ambiguity).
    """
    ref = a_full if a_full is not None else a
    amb_n = ref.group_by("country", "name_core").len().rename({"len": "amb_name"})
    amb_a = ref.group_by("country", "addr_norm", "house_no").len().rename({"len": "amb_addr"})
    a_sel = a.select(A_COLS + ["a_idx"]) if "a_idx" in a.columns else a.select(A_COLS).with_row_index("a_idx")
    a_ctx = (a_sel
               .join(amb_n, on=["country", "name_core"], how="left")
               .join(amb_a, on=["country", "addr_norm", "house_no"], how="left")
               .with_columns(pl.col("amb_name").fill_null(1).cast(pl.Float32),
                             pl.when(pl.col("addr_norm") == "").then(0.0)
                               .otherwise(pl.col("amb_addr").fill_null(1)).cast(pl.Float32).alias("amb_addr")))
    b_ctx = b.select(B_COLS + ["b_idx"]) if "b_idx" in b.columns else b.select(B_COLS).with_row_index("b_idx")
    return a_ctx, b_ctx


def iter_chunks(pairs: pl.DataFrame, a: pl.DataFrame, max_rows: int = 1_500_000) -> Iterator[pl.DataFrame]:
    """Yield pair chunks that keep every Source 1 entity (and its whole partition)
    inside one chunk, so that per-entity and competition features are complete."""
    key = a.select(part=pl.col("country") + "|" + pl.col("state"))
    key = key.with_columns(a["a_idx"]) if "a_idx" in a.columns else key.with_row_index("a_idx")
    p = pairs.join(key, on="a_idx", how="left").with_columns(pl.col("part").fill_null("?"))
    sizes = p.group_by("part").len().sort("len", descending=True)
    groups, cur, cur_n = [], [], 0
    for part, n in zip(sizes["part"], sizes["len"]):
        if cur and cur_n + n > max_rows:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append(part)
        cur_n += n
    if cur:
        groups.append(cur)
    for g in groups:
        yield p.filter(pl.col("part").is_in(g)).drop("part")


def build_features(pairs: pl.DataFrame, a_ctx: pl.DataFrame, b_ctx: pl.DataFrame, pool: Pool,
                   chunk: int = 100_000, n_jobs: int = 8) -> pl.DataFrame:
    """pairs: a_idx, b_idx, sim_name, sim_addr (one partition chunk)."""
    a_r = a_ctx.rename({c: c + "_a" for c in A_COLS})
    b_r = b_ctx.rename({c: c + "_b" for c in B_COLS})
    cols = [c + "_a" for c in STR_COLS] + [c + "_b" for c in STR_COLS]
    mats, outs = [], []
    step = chunk * n_jobs * 2
    # join the string columns slice by slice so that memory stays bounded whatever the chunk size
    for i in range(0, pairs.height, step):
        sl = pairs.slice(i, step).join(a_r, on="a_idx", how="inner").join(b_r, on="b_idx", how="inner")
        tasks = []
        for j in range(0, sl.height, chunk):
            s2 = sl.slice(j, chunk)
            tasks.append(list(zip(*[s2[c].to_list() for c in cols])))
        mats.extend(pool.map(_worker, tasks))
        outs.append(sl.select("a_idx", "b_idx", "sim_name", "sim_addr", pl.col("country_a").alias("country"),
                              "amb_name", "amb_addr",
                              (pl.col("name_script_b") != "latin").cast(pl.Float32).alias("b_nonlatin"),
                              pl.col("name_domain_b").cast(pl.Float32).alias("b_domain"),
                              pl.col("addr_missing_b").cast(pl.Float32).alias("b_addr_missing")))
        del sl
    fm = np.concatenate(mats) if mats else np.zeros((0, len(PAIR_COLS)), np.float32)
    del mats
    out = pl.concat(outs) if outs else pl.DataFrame()
    del outs
    out = out.with_columns([pl.Series(name, fm[:, i]) for i, name in enumerate(PAIR_COLS)])
    del fm
    out = out.with_columns(combo=(pl.col("n_tsr").clip(0, 100) + pl.col("a_tsr").clip(0, 100)) / 2.0)
    out = out.with_columns(
        n_cands=pl.len().over("a_idx").cast(pl.Float32),
        n_tsr_max=pl.col("n_tsr").max().over("a_idx"),
        a_tsr_max=pl.col("a_tsr").max().over("a_idx"),
        combo_max=pl.col("combo").max().over("a_idx"),
        n_tsr_rank=pl.col("n_tsr").rank("min", descending=True).over("a_idx").cast(pl.Float32),
        a_tsr_rank=pl.col("a_tsr").rank("min", descending=True).over("a_idx").cast(pl.Float32),
        combo_rank=pl.col("combo").rank("min", descending=True).over("a_idx").cast(pl.Float32),
        n_hi_name=(pl.col("n_tsr") >= 90).sum().over("a_idx").cast(pl.Float32),
        n_hi_addr=(pl.col("a_tsr") >= 90).sum().over("a_idx").cast(pl.Float32),
    ).with_columns(
        n_tsr_gap=pl.col("n_tsr_max") - pl.col("n_tsr"),
        a_tsr_gap=pl.col("a_tsr_max") - pl.col("a_tsr"),
        combo_gap=pl.col("combo_max") - pl.col("combo"),
    )
    out = out.with_columns(
        b_deg=pl.len().over("b_idx").cast(pl.Float32),
        b_combo_max=pl.col("combo").max().over("b_idx"),
        b_combo_rank=pl.col("combo").rank("min", descending=True).over("b_idx").cast(pl.Float32),
    ).with_columns(b_combo_gap=pl.col("b_combo_max") - pl.col("combo"))
    return out


if __name__ == "__main__":
    import argparse
    import os
    import time

    from block import gt_pairs_idx

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--a", required=True, help="the source1 parquet the pairs were blocked from")
    ap.add_argument("--a-full", default=None, help="full source1 parquet for ambiguity counts")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt", default=None, help="label pairs and subsample negatives")
    ap.add_argument("--neg-frac", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--country", default=None, help="restrict a and b to one country (row indices stay global)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pairs = pl.read_parquet(args.pairs)
    a = pl.read_parquet(args.a).with_row_index("a_idx")
    a_full = pl.read_parquet(args.a_full, columns=["country", "name_core", "addr_norm", "house_no"]) if args.a_full else None
    b = pl.concat([pl.read_parquet(p, columns=["entity_id", "country"] + B_COLS) for p in args.b]).with_row_index("b_idx")
    if args.country:
        a = a.filter(pl.col("country") == args.country)
        b = b.filter(pl.col("country") == args.country)
    a_ctx, b_ctx = context_frames(a, b, a_full)
    truth = None
    if args.gt:
        gt = pl.read_csv(args.gt, separator="\t", quote_char=None, infer_schema=False)
        truth = gt_pairs_idx(gt, a, b).with_columns(y=pl.lit(1, dtype=pl.Int8))
    del a_full, b
    t0 = time.time()
    total = 0
    with Pool(args.jobs) as pool:
        for i, ch in enumerate(iter_chunks(pairs, a)):
            f = build_features(ch, a_ctx, b_ctx, pool, n_jobs=args.jobs)
            if truth is not None:
                f = f.join(truth, on=["a_idx", "b_idx"], how="left").with_columns(pl.col("y").fill_null(0))
                if args.neg_frac < 1.0:
                    keep = (pl.col("y") == 1) | (pl.int_range(pl.len()).shuffle(seed=args.seed + i) < pl.len() * args.neg_frac)
                    f = f.filter(keep)
            f.write_parquet(f"{args.out_dir}/part_{i:03d}.parquet")
            total += f.height
            print(f"  chunk {i}: {ch.height:,} pairs -> {f.height:,} rows kept, t={time.time() - t0:.0f}s", flush=True)
    print(f"features: {total:,} rows x {len(FEATURE_COLS)} features -> {args.out_dir}")
