"""Train the LightGBM pairwise matcher and pick the F_0.5-optimal decision threshold."""
from __future__ import annotations

import glob
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from block import gt_pairs_idx
from features import FEATURE_COLS
from scoring import decide, macro_f05, sweep, truth_from_pairs

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, num_threads=10,
)


def main(feat_dir: str, gt_path: str, out_dir: str, a_path: str, b_paths: list[str], seed: int = 7,
         val_frac: float = 0.2, rounds: int = 3000, neg_frac: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    dirs = feat_dir if isinstance(feat_dir, list) else [feat_dir]
    df = pl.concat([pl.read_parquet(p) for d in dirs for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    a = pl.read_parquet(a_path, columns=["entity_id", "country"])
    b = pl.concat([pl.read_parquet(p, columns=["entity_id"]) for p in b_paths])
    gt = pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
    truth_pairs = gt_pairs_idx(gt, a, b)
    if "y" not in df.columns:
        df = (df.join(truth_pairs.with_columns(y=pl.lit(1, dtype=pl.Int8)), on=["a_idx", "b_idx"], how="left")
                .with_columns(pl.col("y").fill_null(0)))
    rng = np.random.default_rng(seed)
    val_mask = rng.random(a.height) < val_frac
    val_ids = np.flatnonzero(val_mask)
    is_val = df["a_idx"].is_in(pl.Series(val_ids, dtype=pl.UInt32).implode())
    tr, va = df.filter(~is_val), df.filter(is_val)
    del df
    print(f"pairs train={tr.height:,} (pos {tr['y'].sum():,})  val={va.height:,} (pos {va['y'].sum():,})", flush=True)
    X_tr = tr.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_tr = tr["y"].to_numpy()
    del tr
    X_va = va.select(FEATURE_COLS).to_numpy().astype(np.float32)
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_COLS, free_raw_data=True)
    dva = lgb.Dataset(X_va, va["y"].to_numpy(), reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    del X_tr
    p = model.predict(X_va, num_iteration=model.best_iteration)
    scored = va.select(s1_id="a_idx", cand_id="b_idx").with_columns(p=pl.Series(p))
    # scoring set: every validation Source 1 row, singletons included
    truth = truth_from_pairs(truth_pairs.filter(pl.col("a_idx").is_in(pl.Series(val_ids, dtype=pl.UInt32).implode())), val_ids)
    oracle = decide(scored.with_columns(p=va["y"].cast(pl.Float64)), 0.5, one_to_one=False)
    print(f"blocking ceiling (oracle on candidates) macro F0.5 = {macro_f05(oracle, truth):.4f}", flush=True)
    best = (0.5, 0.0, True)
    for one in (False, True):
        for t, f in sweep(scored, truth, one_to_one=one):
            print(f"  one_to_one={one} thr={t:.2f} macroF0.5={f:.4f}", flush=True)
            if f > best[1]:
                best = (t, f, one)
    t0 = best[0]
    for t, f in sweep(scored, truth, thresholds=np.arange(max(0.05, t0 - 0.05), min(0.99, t0 + 0.05), 0.01), one_to_one=best[2]):
        if f > best[1]:
            best = (t, f, best[2])
    print(f"BEST threshold={best[0]:.2f} one_to_one={best[2]} macro F0.5={best[1]:.4f}")
    # per-country breakdown at the chosen operating point
    pred = decide(scored, best[0], best[2])
    ctry = a["country"].to_list()
    for c in sorted(set(ctry)):
        sub = {k: v for k, v in truth.items() if ctry[k] == c}
        print(f"  {c}: macro F0.5 = {macro_f05(pred, sub):.4f} over {len(sub):,} entities")
    model.save_model(os.path.join(out_dir, "model.txt"), num_iteration=model.best_iteration)
    imp = sorted(zip(FEATURE_COLS, model.feature_importance("gain")), key=lambda x: -x[1])
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"threshold": best[0], "one_to_one": best[2], "val_macro_f05": best[1],
                   "best_iteration": model.best_iteration, "neg_frac": neg_frac,
                   "feature_importance": [(k, float(v)) for k, v in imp]}, fh, indent=1)
    print("top features:", [(k, round(v / imp[0][1], 3)) for k, v in imp[:15]])
    scored.write_parquet(os.path.join(out_dir, "val_scored.parquet"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+", help="directories of part_*.parquet")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--a", required=True, help="the sampled source1 parquet used for blocking")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args()
    main(args.features, args.gt, args.out_dir, args.a, args.b, rounds=args.rounds)
