"""Macro-averaged F_0.5 exactly as defined by the challenge, plus the decision
rule that turns pairwise match probabilities into per-entity match lists."""
from __future__ import annotations

import numpy as np
import polars as pl


def f05(prec: float, rec: float) -> float:
    if prec == 0 and rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05(pred: dict[str, set], truth: dict[str, set]) -> float:
    """pred / truth: source1_id -> set of matched ids. Every key of ``truth`` is scored."""
    tot = 0.0
    for s1, t in truth.items():
        p = pred.get(s1, set())
        if not t and not p:
            tot += 1.0
            continue
        if not t or not p:
            continue
        tp = len(p & t)
        tot += f05(tp / len(p), tp / len(t))
    return tot / max(len(truth), 1)


def decide(scored: pl.DataFrame, threshold: float = 0.68, one_to_one: bool = True,
           min_gap: float = 0.0) -> dict[str, set]:
    """scored: s1_id, cand_id, p.  Returns s1_id -> set(cand_id).

    * keep pairs with p >= threshold
    * one_to_one: a Source 2/3 record may be claimed by at most one Source 1 entity
      (the one with the highest probability)
    """
    df = scored.filter(pl.col("p") >= threshold)
    if not df.height:
        return {}
    if one_to_one:
        df = df.sort("p", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)
    out: dict[str, set] = {}
    for s1, c in zip(df["s1_id"].to_list(), df["cand_id"].to_list()):
        out.setdefault(s1, set()).add(c)
    return out


def decide_robust_frame(scored: pl.DataFrame, a_meta: pl.DataFrame, b_meta: pl.DataFrame,
                        threshold: float = 0.67, one_to_one: bool = True,
                        margin: float = 0.03) -> pl.DataFrame:
    """Advanced Multi-Factor Decision Intelligence Engine for Macro F0.5.

    1. Distractor Suppression: Conflicting house numbers penalized (-0.16).
    2. Missing-Address Recovery: Exact names with missing S2/S3 address boosted (+0.12).
    3. House Number Match Boost: Matching street numbers (+0.05).
    4. Exact Compact Name + State Anchor: (+0.04).
    5. Reciprocal Top-1 Match Agreement (Mutual Nearest Neighbors): (+0.03).
    6. Optimal Boundary + 1-to-1 Argmax + Margin Filter.
    """
    if not scored.height:
        return pl.DataFrame({"s1_id": [], "cand_id": []})

    # Select needed columns from metadata
    a_cols = [c for c in ["name_compact", "name_core", "house_no", "state", "addr_missing", "addr_nums"] if c in a_meta.columns]
    b_cols = [c for c in ["name_compact", "name_core", "house_no", "state", "addr_missing", "addr_nums"] if c in b_meta.columns]

    a_m = a_meta.select(["a_idx"] + a_cols).rename({"a_idx": "s1_id", **{c: c + "_a" for c in a_cols}})
    b_m = b_meta.select(["b_idx"] + b_cols).rename({"b_idx": "cand_id", **{c: c + "_b" for c in b_cols}})

    df = (scored
          .join(a_m, on="s1_id", how="inner")
          .join(b_m, on="cand_id", how="inner"))

    # Conditions
    # 1. House number relations
    has_house_a = (pl.col("house_no_a") != "")
    has_house_b = (pl.col("house_no_b") != "")
    is_house_match = has_house_a & has_house_b & (pl.col("house_no_a") == pl.col("house_no_b"))
    is_house_conflict = has_house_a & has_house_b & (pl.col("house_no_a") != pl.col("house_no_b"))

    # 2. Exact compact name agreement
    has_name = (pl.col("name_compact_a") != "") & (pl.col("name_compact_b") != "")
    is_exact_name = has_name & (pl.col("name_compact_a") == pl.col("name_compact_b"))
    is_same_state = (pl.col("state_a") == "") | (pl.col("state_b") == "") | (pl.col("state_a") == pl.col("state_b"))

    # 3. Missing address recovery in S2/S3 (addresses the 41% false negatives)
    is_b_missing_addr = (pl.col("addr_missing_b") == 1) if "addr_missing_b" in df.columns else pl.lit(False)
    is_name_match_no_addr = is_b_missing_addr & is_exact_name & is_same_state

    # 4. Reciprocal best match (mutual rank 1)
    df = df.with_columns(
        rank_for_s1=pl.col("p").rank("min", descending=True).over("s1_id"),
        rank_for_cand=pl.col("p").rank("min", descending=True).over("cand_id")
    )
    is_reciprocal_best = (pl.col("rank_for_s1") == 1) & (pl.col("rank_for_cand") == 1)

    # Effective probability calculation
    p_eff_expr = (
        pl.col("p")
        - pl.when(is_house_conflict).then(0.16).otherwise(0.0)
        + pl.when(is_house_match).then(0.05).otherwise(0.0)
        + pl.when(is_exact_name & is_same_state).then(0.04).otherwise(0.0)
        + pl.when(is_name_match_no_addr).then(0.12).otherwise(0.0)
        + pl.when(is_reciprocal_best).then(0.03).otherwise(0.0)
    ).clip(0.0, 1.0)

    df = df.with_columns(p_eff=p_eff_expr)

    # Filter by threshold
    df = df.filter(pl.col("p_eff") >= threshold)
    if not df.height:
        return pl.DataFrame({"s1_id": [], "cand_id": []})

    # 1-to-1 argmax per candidate
    if one_to_one:
        df = df.sort("p_eff", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)

    # Margin check per S1 entity
    if margin > 0.0:
        df = df.with_columns(
            p_max=pl.col("p_eff").max().over("s1_id"),
            n_cands=pl.len().over("s1_id")
        ).with_columns(
            p_gap=pl.col("p_max") - pl.col("p_eff")
        ).filter(
            (pl.col("n_cands") == 1) | (pl.col("p_gap") <= margin) | (pl.col("p_eff") >= threshold + 0.06)
        )

    return df.select("s1_id", "cand_id")


def truth_from_pairs(gp: pl.DataFrame, all_a) -> dict:
    """gp: (a_idx, b_idx) true pairs; all_a: iterable of a_idx to score (singletons included)."""
    out = {int(x): set() for x in all_a}
    for x, y in zip(gp["a_idx"].to_list(), gp["b_idx"].to_list()):
        out.setdefault(int(x), set()).add(int(y))
    return out


def truth_from_gt(gt: pl.DataFrame, s1_ids) -> dict[str, set]:
    g = gt.filter(pl.col("source1_entity_id").is_in(s1_ids)).fill_null("")
    return {s: set(x for x in m.split(",") if x) for s, m in zip(g["source1_entity_id"], g["matched_entity_ids"])}


def sweep(scored: pl.DataFrame, truth: dict[str, set], thresholds=None, one_to_one=True) -> list[tuple[float, float]]:
    thresholds = thresholds if thresholds is not None else np.arange(0.20, 0.96, 0.05)
    res = []
    for t in thresholds:
        res.append((float(t), macro_f05(decide(scored, float(t), one_to_one), truth)))
    return res
