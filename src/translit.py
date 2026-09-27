"""Learn a word-level transliteration dictionary (Indic scripts -> Latin) from the
training ground truth. For every matched (S1, S2/S3) pair whose S2/S3 name is
written in a non-Latin script, the Latin S1 name and the Indic name are aligned
token-by-token when they have the same number of tokens. The most frequent Latin
token for each Indic token is kept. Nothing external is used.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl

ASCII_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~]+")


def _script_of(tok: str) -> str:
    for ch in tok:
        if ch.isalpha():
            try:
                n = unicodedata.name(ch).split()[0]
            except ValueError:
                continue
            return "latin" if n == "LATIN" else n
    return "latin"


def _is_latin(tok: str) -> bool:
    return tok.isascii()


def _clean_latin(name: str) -> list[str]:
    from unidecode import unidecode
    t = unidecode(name).lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return t.split()


def _clean_indic(name: str) -> list[str]:
    t = ASCII_PUNCT_RE.sub(" ", name)
    return [w for w in t.split() if _script_of(w) != "latin"]


def build(train_dir: str, out_path: str, min_count: int = 2, min_share: float = 0.5) -> None:
    s1 = pl.read_csv(f"{train_dir}/train_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
    gt = pl.read_csv(f"{train_dir}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    s2 = pl.read_csv(f"{train_dir}/train_source2.tsv", separator="\t", quote_char=None, infer_schema=False)
    s3 = pl.read_csv(f"{train_dir}/train_source3.tsv", separator="\t", quote_char=None, infer_schema=False)
    other = pl.concat([s2, s3]).select("entity_id", "business_name")
    other = other.filter(pl.col("business_name").str.contains(r"[\u0900-\u0DFF]"))
    pairs = (gt.fill_null("").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"matched_entity_ids": "entity_id"})
             .join(other, on="entity_id", how="inner")
             .join(s1.select("entity_id", "business_name").rename({"entity_id": "source1_entity_id",
                                                                    "business_name": "s1_name"}),
                   on="source1_entity_id", how="inner"))
    counts: dict[str, Counter] = defaultdict(Counter)
    n_aligned = 0
    for s1_name, ind_name in zip(pairs["s1_name"], pairs["business_name"]):
        lat = _clean_latin(s1_name)
        ind = _clean_indic(ind_name)
        if not ind:
            continue
        # Indic names sometimes mix Latin legal suffixes; align only the Indic-script tokens
        # against the Latin tokens when counts agree, otherwise against the prefix of same length.
        if len(ind) == len(lat):
            n_aligned += 1
            for a, b in zip(ind, lat):
                counts[a][b] += 1
        elif len(ind) < len(lat):
            # Indic tokens correspond to the leading Latin tokens (legal suffix often stays Latin)
            for a, b in zip(ind, lat[: len(ind)]):
                counts[a][b] += 0.5
    table = {}
    for ind, c in counts.items():
        tok, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            table[ind] = tok
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False)
    print(f"pairs with Indic names: {pairs.height:,}; aligned: {n_aligned:,}; dictionary size: {len(table):,}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.train_dir, a.out)
