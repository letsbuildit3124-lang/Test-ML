import os
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool

sys.path.insert(0, "src")
from normalize import normalize_record, load_translit
from block import run_blocking
from features import context_frames, build_features, FEATURE_COLS
from scoring import macro_f05, f05, decide

load_translit("work/translit.json")

def evaluate_predictions(pred_dict, gt_dict):
    prec_list, rec_list, f05_list = [], [], []
    for s1, truth in gt_dict.items():
        preds = pred_dict.get(s1, set())
        if not truth and not preds:
            f05_list.append(1.0)
            continue
        if not truth or not preds:
            f05_list.append(0.0)
            continue
        tp = len(preds & truth)
        p = tp / len(preds)
        r = tp / len(truth)
        f05_val = f05(p, r)
        f05_list.append(f05_val)
        prec_list.append(p)
        rec_list.append(r)
    return float(np.mean(f05_list)), float(np.mean(prec_list) if prec_list else 0), float(np.mean(rec_list) if rec_list else 0)


def main():
    print("=== FAST 20,000 VALIDATION EXPERIMENT (PARALLEL 8 JOBS) ===")
    t0 = time.time()
    gt_all = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    gt_sample = gt_all.head(20000)
    val_s1_ids = set(gt_sample["source1_entity_id"].to_list())

    s1_raw = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
    s1_val = s1_raw.filter(pl.col("entity_id").is_in(list(val_s1_ids)))

    val_matches = set()
    for m in gt_sample["matched_entity_ids"].to_list():
        if m:
            for x in m.split(","):
                if x:
                    val_matches.add(x)

    val_countries = set(s1_val["country"].to_list())
    s2_raw = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", quote_char=None, infer_schema=False)
    s3_raw = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", quote_char=None, infer_schema=False)
    s2_sub = s2_raw.filter(pl.col("country").is_in(list(val_countries))).head(100000)
    s3_sub = s3_raw.filter(pl.col("country").is_in(list(val_countries))).head(100000)
    s2_matched = s2_raw.filter(pl.col("entity_id").is_in(list(val_matches)))
    s3_matched = s3_raw.filter(pl.col("entity_id").is_in(list(val_matches)))
    b_raw = pl.concat([s2_sub, s3_sub, s2_matched, s3_matched]).unique("entity_id")

    s1_norm = pl.DataFrame([normalize_record(*r) for r in zip(s1_val["entity_id"], s1_val["business_name"], s1_val["business_address"], s1_val["country"])])
    b_norm = pl.DataFrame([normalize_record(*r) for r in zip(b_raw["entity_id"], b_raw["business_name"], b_raw["business_address"], b_raw["country"])])
    s1_norm = s1_norm.with_columns(pl.col("name_domain").cast(pl.Int8), pl.col("addr_missing").cast(pl.Int8))
    b_norm = b_norm.with_columns(pl.col("name_domain").cast(pl.Int8), pl.col("addr_missing").cast(pl.Int8))

    gt_dict = {}
    for s1, m in zip(gt_sample["source1_entity_id"], gt_sample["matched_entity_ids"]):
        gt_dict[s1] = set(x for x in (m or "").split(",") if x)

    # 1. Blocking
    pairs_base = run_blocking(s1_norm, b_norm, threads=8, verbose=False)
    ai = s1_norm.select("entity_id").with_row_index("a_idx")
    bi = b_norm.select("entity_id").with_row_index("b_idx")

    # 2. Parallel Features
    a_ctx, b_ctx = context_frames(s1_norm, b_norm)
    with Pool(8) as pool:
        f_val = build_features(pairs_base, a_ctx, b_ctx, pool, n_jobs=8)

    # 3. Model Scoring
    model = lgb.Booster(model_file="work/model/model.txt")
    feat_mat = f_val.select(FEATURE_COLS).to_numpy()
    p_scores = model.predict(feat_mat, num_threads=8)

    scored_df = f_val.select(
        s1_idx="a_idx", cand_idx="b_idx",
        combo="combo", house_rel="house_rel", amb_name="amb_name",
        sim_name="sim_name", sim_addr="sim_addr"
    ).with_columns(
        p=pl.Series(p_scores, dtype=pl.Float32)
    ).join(ai.rename({"a_idx": "s1_idx", "entity_id": "s1_id"}), on="s1_idx"
    ).join(bi.rename({"b_idx": "cand_idx", "entity_id": "cand_id"}), on="cand_idx")

    print(f"Scored {scored_df.height:,} candidate pairs in {time.time() - t0:.1f}s\n")

    # 4. Base Threshold Sweep
    print("--- 1. BASELINE THRESHOLD SWEEP ---")
    best_base_thr = 0.68
    best_base_f05 = 0.0
    for thr in np.arange(0.55, 0.82, 0.01):
        pred = decide(scored_df.select("s1_id", "cand_id", "p"), thr, one_to_one=True)
        score, pr, rc = evaluate_predictions(pred, gt_dict)
        if score > best_base_f05:
            best_base_f05 = score
            best_base_thr = thr
        if abs(thr - 0.68) < 0.005 or abs(thr - 0.65) < 0.005 or abs(thr - 0.70) < 0.005:
            print(f"  Threshold {thr:.2f} -> Macro F0.5: {score:.4f} (Prec: {pr:.4f}, Rec: {rc:.4f})")

    print(f"  >>> Best Baseline: Threshold {best_base_thr:.2f} -> Macro F0.5 = {best_base_f05:.4f}\n")

    # 5. Advanced Decision Rules (Margin + House Number Conflict Penalty + Ambiguity Filter)
    print("--- 2. ADVANCED DECISION RULES GRID SWEEP ---")
    best_rule_score = best_base_f05
    best_config = {}

    for base_thr in np.arange(0.58, 0.76, 0.02):
        for house_conflict_penalty in [0.0, 0.05, 0.10, 0.15]:
            for margin in [0.0, 0.02, 0.05, 0.08]:
                # Apply adjusted probabilities
                adj_df = scored_df.with_columns(
                    p_adj=pl.when(pl.col("house_rel") == -1.0)
                            .then(pl.col("p") - house_conflict_penalty)
                            .otherwise(pl.col("p"))
                ).filter(pl.col("p_adj") >= base_thr)

                if not adj_df.height:
                    continue

                # 1-to-1 greedy
                df_one = adj_df.sort("p_adj", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)

                # Margin rule on S1 groups
                df_sorted = df_one.sort(["s1_id", "p_adj"], descending=[False, True])
                s1_groups = {}
                for s1, c, p, amb in zip(df_sorted["s1_id"], df_sorted["cand_id"], df_sorted["p_adj"], df_sorted["amb_name"]):
                    s1_groups.setdefault(s1, []).append((c, p, amb))

                pred = {}
                for s1, cands in s1_groups.items():
                    if len(cands) == 1:
                        c, p, amb = cands[0]
                        # If single candidate has high ambiguity and marginal score, check threshold
                        pred[s1] = {c}
                    else:
                        top_c, top_p, top_amb = cands[0]
                        sec_c, sec_p, sec_amb = cands[1]
                        if margin <= 0.0 or (top_p - sec_p) >= margin or top_p >= base_thr + 0.08:
                            pred[s1] = {top_c}
                        else:
                            # Keep both if high confidence
                            pred[s1] = {top_c, sec_c}

                score, pr, rc = evaluate_predictions(pred, gt_dict)
                if score > best_rule_score:
                    best_rule_score = score
                    best_config = {
                        "threshold": float(base_thr),
                        "house_conflict_penalty": float(house_conflict_penalty),
                        "margin": float(margin),
                        "score": float(score),
                        "precision": float(pr),
                        "recall": float(rc)
                    }

    print(f"  >>> OPTIMAL BOOSTED CONFIGURATION:")
    print(f"      Threshold: {best_config.get('threshold', best_base_thr):.2f}")
    print(f"      House Conflict Penalty: {best_config.get('house_conflict_penalty', 0.0):.2f}")
    print(f"      Margin Rule: {best_config.get('margin', 0.0):.2f}")
    print(f"      Validation Macro F0.5: {best_config.get('score', best_base_f05):.4f} (Prec: {best_config.get('precision', 0):.4f}, Rec: {best_config.get('recall', 0):.4f})")
    print(f"      Score Delta: +{(best_config.get('score', best_base_f05) - best_base_f05)*100:.2f} percentage points!")

if __name__ == "__main__":
    main()
