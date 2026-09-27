"""Block the full training Source 1 against Source 2+3, one country at a time, global row indices."""
import sys, time
import polars as pl
sys.path.insert(0, 'code/business_entity_resolution/src')
from block import run_blocking, evaluate_blocking
country = sys.argv[1]
cols = ["entity_id", "country", "state", "name_compact", "addr_norm", "addr_nums"]
a = pl.read_parquet('work/norm/train_source1.parquet', columns=cols).with_row_index("a_idx").filter(pl.col("country") == country)
b = pl.concat([pl.read_parquet(f'work/norm/train_source{i}.parquet', columns=cols) for i in (2, 3)]).with_row_index("b_idx").filter(pl.col("country") == country)
print(f"[{country}] A={a.height:,} B={b.height:,}", flush=True)
t0 = time.time()
pairs = run_blocking(a.drop("a_idx"), b.drop("b_idx"), threads=8, verbose=False)
gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator="\t", quote_char=None, infer_schema=False)
evaluate_blocking(pairs, gt, a.drop("a_idx"), b.drop("b_idx"))
ag, bg = a["a_idx"].to_numpy(), b["b_idx"].to_numpy()
pairs = pairs.with_columns(a_idx=pl.Series(ag[pairs["a_idx"].to_numpy()], dtype=pl.UInt32),
                           b_idx=pl.Series(bg[pairs["b_idx"].to_numpy()], dtype=pl.UInt32))
pairs.write_parquet(f'work/full_pairs_{country}.parquet')
print(f"[{country}] wrote {pairs.height:,} pairs in {time.time() - t0:.0f}s", flush=True)
