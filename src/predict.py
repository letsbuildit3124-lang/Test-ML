"""End-to-end inference on the test set.

Every stage runs once per country in its own process so that no step ever holds
more than a fraction of the data (the pipeline is designed for a 16 GB laptop):

  block  : normalised parquet -> candidate pairs        work/test_pairs_<country>.parquet
  score  : pairs -> features -> LightGBM probabilities  work/test_scored_<country>.parquet
  output : threshold + one-to-one assignment            output/matching_results.tsv
                                                          output/candidate_pairs.tsv
Row indices (a_idx, b_idx) are global positions in test_source1 and in the
concatenation test_source2 + test_source3, so the per-country parts are compatible.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

BLOCK_COLS = ["country", "state", "name_compact", "addr_norm", "addr_nums"]


def _scan(norm_dir: str, which: str) -> pl.LazyFrame:
    if which == "a":
        return pl.scan_parquet(f"{norm_dir}/test_source1.parquet").with_row_index("a_idx")
    return pl.concat([pl.scan_parquet(f"{norm_dir}/test_source2.parquet"),
                      pl.scan_parquet(f"{norm_dir}/test_source3.parquet")]).with_row_index("b_idx")


def _load(norm_dir: str, which: str, columns: list[str], country: str | None) -> pl.DataFrame:
    lf = _scan(norm_dir, which)
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    idx = "a_idx" if which == "a" else "b_idx"
    return lf.select([idx] + columns).collect()


def countries(norm_dir: str) -> list[str]:
    return sorted(pl.scan_parquet(f"{norm_dir}/test_source1.parquet").select("country").unique().collect()["country"].to_list())


def stage_block(norm_dir: str, work_dir: str, threads: int, country: str) -> None:
    from block import run_blocking
    t0 = time.time()
    a = _load(norm_dir, "a", BLOCK_COLS, country)
    b = _load(norm_dir, "b", BLOCK_COLS, country)
    print(f"[{country}] A={a.height:,} B={b.height:,}", flush=True)
    pairs = run_blocking(a, b, threads=threads)
    a_glob, b_glob = a["a_idx"].to_numpy(), b["b_idx"].to_numpy()
    pairs = pairs.with_columns(a_idx=pl.Series(a_glob[pairs["a_idx"].to_numpy()], dtype=pl.UInt32),
                               b_idx=pl.Series(b_glob[pairs["b_idx"].to_numpy()], dtype=pl.UInt32))
    pairs.write_parquet(f"{work_dir}/test_pairs_{country}.parquet")
    print(f"[{country}] blocking: {pairs.height:,} pairs ({pairs.height / a.height:.1f}/entity) in {time.time() - t0:.0f}s", flush=True)


def stage_score(norm_dir: str, model_dir: str, work_dir: str, n_jobs: int, threads: int, country: str) -> None:
    import lightgbm as lgb
    from features import A_COLS, B_COLS, FEATURE_COLS, build_features, context_frames, iter_chunks
    t0 = time.time()
    a = _load(norm_dir, "a", A_COLS, country)
    b = _load(norm_dir, "b", B_COLS, country)
    a_ctx, b_ctx = context_frames(a, b)
    a_key = a.select("a_idx", "country", "state")
    del a, b
    pairs = pl.read_parquet(f"{work_dir}/test_pairs_{country}.parquet")
    model = lgb.Booster(model_file=f"{model_dir}/model.txt")
    parts = []
    with Pool(n_jobs) as pool:
        for i, ch in enumerate(iter_chunks(pairs, a_key)):
            ck = f"{work_dir}/test_scored_{country}_chunk{i:03d}.parquet"
            if os.path.exists(ck):  # checkpoint from an interrupted run
                parts.append(pl.read_parquet(ck))
                print(f"[{country}] chunk {i}: reusing checkpoint", flush=True)
                continue
            f = build_features(ch, a_ctx, b_ctx, pool, n_jobs=n_jobs)
            p = model.predict(f.select(FEATURE_COLS).to_numpy(), num_threads=threads)
            part = f.select("a_idx", "b_idx").with_columns(p=pl.Series(p, dtype=pl.Float32))
            part.write_parquet(ck)
            parts.append(part)
            print(f"[{country}] chunk {i}: {ch.height:,} pairs scored, t={time.time() - t0:.0f}s", flush=True)
            del f
    scored = pl.concat(parts)
    scored.write_parquet(f"{work_dir}/test_scored_{country}.parquet")
    for ck in glob.glob(f"{work_dir}/test_scored_{country}_chunk*.parquet"):
        os.remove(ck)
    print(f"[{country}] scored {scored.height:,} pairs in {time.time() - t0:.0f}s", flush=True)


def _lists(df: pl.DataFrame, a_ids: pl.DataFrame, b_ids: pl.DataFrame, col: str) -> pl.DataFrame:
    g = (df.join(b_ids, on="b_idx").group_by("a_idx")
           .agg(pl.col("cand").sort().str.join(",").alias(col)))
    return (a_ids.join(g, on="a_idx", how="left").with_columns(pl.col(col).fill_null(""))
                 .sort("a_idx").select(pl.col("entity_id").alias("source1_entity_id"), col))


def stage_output(norm_dir: str, model_dir: str, work_dir: str, country: str) -> None:
    from scoring import decide_frame
    cfg = json.load(open(f"{model_dir}/config.json"))
    a_ids = _load(norm_dir, "a", ["entity_id"], country)
    b_ids = _load(norm_dir, "b", ["entity_id"], country).rename({"entity_id": "cand"})
    scored = pl.read_parquet(f"{work_dir}/test_scored_{country}.parquet").rename({"a_idx": "s1_id", "b_idx": "cand_id"})
    matches = decide_frame(scored, cfg["threshold"], cfg["one_to_one"]).rename({"s1_id": "a_idx", "cand_id": "b_idx"})
    del scored
    m = _lists(matches, a_ids, b_ids, "matched_entity_ids")
    m.write_csv(f"{work_dir}/out_matches_{country}.tsv", separator="\t", quote_style="never")
    n_ent = (m["matched_entity_ids"] != "").sum()
    print(f"[{country}] matches: {matches.height:,} pairs for {n_ent:,} entities; {m.height - n_ent:,} singletons", flush=True)
    del matches, m
    pairs = pl.read_parquet(f"{work_dir}/test_pairs_{country}.parquet", columns=["a_idx", "b_idx"])
    c = _lists(pairs, a_ids, b_ids, "candidate_entity_ids")
    c.write_csv(f"{work_dir}/out_cands_{country}.tsv", separator="\t", quote_style="never")
    print(f"[{country}] candidates: {pairs.height:,} pairs written", flush=True)


def merge_outputs(work_dir: str, out_dir: str, cs: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for kind, name in (("matches", "matching_results.tsv"), ("cands", "candidate_pairs.tsv")):
        with open(f"{out_dir}/{name}", "w", encoding="utf-8") as out:
            for i, c in enumerate(cs):
                with open(f"{work_dir}/out_{kind}_{c}.tsv", encoding="utf-8") as fh:
                    header = fh.readline()
                    if i == 0:
                        out.write(header)
                    for line in fh:
                        out.write(line)
    print(f"merged outputs -> {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "block", "score", "output"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true", help="reuse per-country parts already on disk")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    if args.stage == "all":
        cs = countries(args.norm_dir)
        for st in ("block", "score", "output"):
            for c in cs:
                marker = {"block": f"test_pairs_{c}.parquet", "score": f"test_scored_{c}.parquet",
                          "output": f"out_cands_{c}.tsv"}[st]
                if args.skip_existing and os.path.exists(f"{args.work_dir}/{marker}"):
                    print(f"=== stage {st} {c}: reusing {marker}", flush=True)
                    continue
                print(f"=== stage {st} {c} {time.strftime('%H:%M:%S')}", flush=True)
                cmd = [sys.executable, __file__, "--stage", st, "--country", c, "--norm-dir", args.norm_dir,
                       "--model-dir", args.model_dir, "--out-dir", args.out_dir, "--work-dir", args.work_dir,
                       "--jobs", str(args.jobs), "--threads", str(args.threads)]
                subprocess.run(cmd, check=True)
        merge_outputs(args.work_dir, args.out_dir, cs)
        print(f"=== done {time.strftime('%H:%M:%S')}", flush=True)
    elif args.stage == "block":
        stage_block(args.norm_dir, args.work_dir, args.threads, args.country)
    elif args.stage == "score":
        stage_score(args.norm_dir, args.model_dir, args.work_dir, args.jobs, args.threads, args.country)
    else:
        stage_output(args.norm_dir, args.model_dir, args.work_dir, args.country)
