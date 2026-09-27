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
                        threshold: float = 0.68, one_to_one: bool = True,
                        margin: float = 0.03) -> pl.DataFrame:
    """Multi-Layer Robust Decision Engine for Maximizing Macro F0.5.

    1. Joins metadata to inspect house numbers, exact compact names, and states.
    2. Layer A (Distractor Rejection): Conflicting house numbers penalized (p_eff = p - 0.14).
    3. Layer B (High-Confidence Agreement): Exact compact names & matching states boosted (p_eff = p + 0.04).
    4. Layer C (Optimal Boundary): Filters p_eff >= threshold.
    5. Layer D (1-to-1 Argmax): Assigns candidate uniquely to best S1 entity.
    6. Layer E (Confidence Margin): Protects singletons from marginal ambiguous matches.
    """
    if not scored.height:
        return pl.DataFrame({"s1_id": [], "cand_id": []})

    # Join metadata
    a_m = a_meta.select(s1_id="a_idx", name_cmp_a="name_compact", house_a="house_no", st_a="state")
    b_m = b_meta.select(cand_id="b_idx", name_cmp_b="name_compact", house_b="house_no", st_b="state")

    df = (scored
          .join(a_m, on="s1_id", how="inner")
          .join(b_m, on="cand_id", how="inner"))

    # Calculate effective probability
    # If house numbers both exist and conflict: penalty of 0.14
    is_house_conflict = (pl.col("house_a") != "") & (pl.col("house_b") != "") & (pl.col("house_a") != pl.col("house_b"))
    # If exact compact name and same state: boost of 0.04
    is_exact_name_state = (pl.col("name_cmp_a") != "") & (pl.col("name_cmp_a") == pl.col("name_cmp_b")) & ((pl.col("st_a") == "") | (pl.col("st_a") == pl.col("st_b")))

    df = df.with_columns(
        p_eff=(pl.col("p")
               - pl.when(is_house_conflict).then(0.14).otherwise(0.0)
               + pl.when(is_exact_name_state).then(0.04).otherwise(0.0))
    )

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
