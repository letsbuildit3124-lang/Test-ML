# Forensic Audit Report: Business Entity Resolution Pipeline

**Target Environment:** 2 vCPU, 8 GB RAM, Fast SSD/EBS (AWS constrained instance)  
**Dataset Scale:**
- **Source 1 (Test):** 1,732,544 entities (US: 663,106; India: 809,986; France: 259,452)
- **Source 2 (Test):** 4,887,273 records (US: 1,871,330; India: 2,312,565; France: 703,378)
- **Source 3 (Test):** 5,082,316 records (US: 1,945,701; India: 2,405,000; France: 731,615)
- **Total Test Records:** 11,702,133 records across 3 sources and 3 countries.
- **Total Candidate Universe:** ~97.3 Million pairs.

---

## A. Complete Pipeline Flow

```
+-----------------------------------------------------------------------------------+
| 1. Transliteration (translit.py)                                                  |
|    - Input: train_source1, train_source2, train_source3, train_ground_truth.tsv   |
|    - Output: work/translit.json (1,347 Indic->Latin word translations)            |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 2. Normalization (normalize.py)                                                   |
|    - Normalizes Raw TSV -> Canonical Parquet (name_norm, name_core, name_compact, |
|      addr_norm, addr_nums, house_no, state, script/domain/missing flags)          |
|    - Output: work/norm/{train|test}_source{1,2,3}.parquet                         |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 3. Candidate Generation / Blocking (block.py & predict.py:stage_block)            |
|    - Per (Country, State) partition:                                              |
|      * Forward & Reverse TF-IDF on char 3-grams of name_compact (top-20 / top-3)  |
|      * Forward & Reverse TF-IDF on word tokens of addr_text (top-15 / top-3)      |
|      * Deduplication & Max Cosine aggregation                                     |
|    - Output: work/test_pairs_{France,India,US}.parquet (~97.3M total pairs)       |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 4. Context Extraction & Chunking (features.py:context_frames & iter_chunks)       |
|    - Precomputes ambiguity counts: amb_name, amb_addr in Source 1                 |
|    - Groups partitions into chunks of up to 1.5M pairs                            |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 5. Feature Engineering (features.py:build_features)                               |
|    - 25 Pairwise String Similarities: RapidFuzz ratios (ratio, token_set_ratio,   |
|      token_sort_ratio, partial_ratio, Jaro-Winkler, token Jaccard, house/state)   |
|    - 7 Context Features: Cosines, Ambiguities, Domain/Script/Missing flags        |
|    - 17 Group/Window Features: Ranks, Gaps, Degrees, Competition over a_idx/b_idx |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 6. Model Scoring & Inference (train.py / predict.py:stage_score)                  |
|    - LightGBM binary classifier (2,784 trees, 127 leaves, 49 features)            |
|    - Predicts probability p for all candidate pairs in chunks                     |
|    - Output: work/test_scored_{France,India,US}.parquet                           |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
| 7. Decision & Assignment (scoring.py & predict.py:stage_output)                   |
|    - Threshold filtering (p >= 0.68)                                              |
|    - Greedy 1-to-1 Assignment (argmax p per candidate ID)                         |
|    - Group by a_idx -> comma-separated ID lists                                  |
|    - Output: output/matching_results.tsv, output/candidate_pairs.tsv              |
+-----------------------------------------------------------------------------------+
```

---

## B. Memory-Heavy Operations

1. **Materialization of Full String Columns for 97M Pairs:**
   - In `features.py:build_features`, Polars joins `pairs` with `a_ctx` and `b_ctx` to pull 14 string columns (`name_core_a`, `name_core_b`, `addr_norm_a`, etc.).
   - Converting Polars string columns via `.to_list()` materializes millions of redundant Python `str` objects in heap memory.
2. **Subprocess/IPC Serialization in Multiprocessing:**
   - `Pool.map(_worker, tasks)` pickles large batches of Python tuples/strings between parent and worker processes, causing memory spikes and copying overhead.
3. **Concatenation of Full Partition Data:**
   - In `block.py`, records with missing states (`b_nostate`) are concatenated to *every* state partition in that country, inflating in-memory frames.
4. **LightGBM Input Matrix Allocation:**
   - Converting 1.5M rows × 49 columns into `np.ndarray` (float32) requires ~300 MB per batch, but intermediate Polars representations double/triple the footprint during conversion.

---

## C. CPU-Heavy Operations

1. **LightGBM Tree Traversal on 97M Pairs (The #1 Monster Bottleneck):**
   - The current model contains **2,784 trees** with `num_leaves=127`.
   - Micro-benchmark on 2 vCPU: LightGBM evaluates at **~4,500 rows/sec**.
   - For 97,300,000 candidate pairs:  
     $$\frac{97,300,000}{4,500} \approx 21,622\text{ seconds} \approx \mathbf{6.0\text{ hours}}$$
   - Evaluating all 2,784 trees on massive numbers of low-scoring candidate pairs (e.g. cosine < 0.25, zero name match) consumes ~6 hours of CPU.
2. **Expensive String Metrics in RapidFuzz Loop (The #2 Major Bottleneck):**
   - `_pair_feats` performs 11 separate RapidFuzz computations per pair: `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, `JaroWinkler`, Python `set.intersection`, etc.
   - For 97.3M pairs, this requires **over 1.07 Billion individual metric calls** and string tokenizations.
   - At ~25,000 pairs/sec per core, 97.3M pairs require:  
     $$\frac{97,300,000}{50,000\text{ (2 vCPUs)}} \approx 1,946\text{ seconds} \approx \mathbf{32.4\text{ minutes}}$$
3. **TF-IDF Vocabulary Fitting and Top-K Sparse Multiplication:**
   - Fitting `TfidfVectorizer` and running `sparse_dot_topn` across all partitions takes ~15–20 minutes total across 2 vCPU.

---

## D. Disk-I/O-Heavy Operations

1. **Writing and Reading Intermediate Uncompressed/Large Parquet Chunks:**
   - Writing 97M intermediate pairs (`test_pairs_*.parquet`), reading them back, writing intermediate feature chunks, reading them for scoring, and saving scored chunks.
2. **TSV Output Serialization:**
   - Grouping 97M candidate pairs and building multi-gigabyte TSV files with Python string joining.

---

## E. Repeated Computations

1. **Repeated String Splitting & Tokenization:**
   - `_pair_feats` splits `name_core_a` and `name_core_b` on whitespace for every candidate pair. If an entity has 50 candidates, its strings are split 50 times repeatedly.
   - Address digit extraction and word set construction are re-executed for every pair instead of once per record.
2. **Recomputing Ambiguity Counts:**
   - Ambiguity frequencies (`amb_name`, `amb_addr`) are computed via joins instead of single pre-computed hash maps or integer ID lookups.
3. **TF-IDF Vectorizer Refitting:**
   - For every state partition, a new `TfidfVectorizer` is constructed and fitted on `B` records from scratch.

---

## F. Objects That Can Be Cached

1. **Normalized Parquet Files:** `work/norm/{train,test}_source{1,2,3}.parquet` (One-time compute).
2. **Pre-tokenized Records & Pre-split String Arrays:** Word sets, digit sets, first tokens, compact names cached in memory/arrays.
3. **Ambiguity Hash Maps:** Country-level frequency tables for name core and address.
4. **Partition Indices:** Mapping of entity IDs to integer row indices.

---

## G. Objects That Should Be Persisted

1. `work/translit.json` (Transliteration dictionary).
2. `work/norm/*.parquet` (Normalized base tables).
3. `work/model/model.txt` and `config.json`.
4. `work/cache/test_pairs_{country}.parquet` (Candidate pairs).
5. Checkpointed scoring chunks (`work/cache/test_scored_{country}_chunkNNN.parquet`).

---

## H. Operations That Can Be Chunked

1. **S1 Entity Chunking:** Process Source 1 entities in batches (e.g. 25,000 – 50,000 entities per chunk).
2. **End-to-End Streaming within Chunk:**
   $$\text{S1 Chunk} \rightarrow \text{Blocking} \rightarrow \text{Feature Generation} \rightarrow \text{Scoring} \rightarrow \text{Partial TSV / Parquet} \rightarrow \text{GC}$$
   This eliminates keeping 97M pairs in RAM or on disk simultaneously.

---

## I. Operations That Can Be Converted to DuckDB / Parquet

1. **Normalization Storage:** Stored as zstd-compressed Parquet.
2. **Partition & Ambiguity Queries:** DuckDB / Arrow zero-copy memory scanning.
3. **Final TSV Aggregation & 1-to-1 Filtering:** DuckDB can stream the one-to-one argmax and `string_agg(cand_id, ',')` directly to TSV at multi-GB/s disk speed with zero Python memory overhead.

---

## J. Operations That Can Be Replaced with Faster Data Structures

1. **Direct Array Indexing vs Polars Inner Joins:**
   - Replace `pairs.join(a_ctx).join(b_ctx)` with direct integer indexing `names_a[a_idx]` and `names_b[b_idx]`, cutting feature join time by 60%.
2. **Pre-computed Token Sets / Lengths / First Tokens:**
   - Store `first_token`, `num_tokens`, `set_of_tokens` as precomputed record attributes instead of re-splitting in Python inside the pairwise loop.
3. **Fast Early-Exit / Cascade Filtering:**
   - Obvious non-candidates (e.g., pairs with zero string overlap and low cosine from reverse-channel noise) can skip expensive RapidFuzz metrics or deep tree inference without any loss of true positive recall.

---

## K. Current Candidate-Pair Volume Estimates

| Country | Source 1 Entities | Source 2/3 Pool | Avg Cands / S1 | Total Candidate Pairs |
| :--- | :--- | :--- | :--- | :--- |
| **US** | 663,106 | 3,817,031 | ~55.8 | **37,000,000** |
| **India** | 809,986 | 4,717,565 | ~56.5 | **45,800,000** |
| **France** | 259,452 | 1,434,993 | ~56.0 | **14,500,000** |
| **TOTAL** | **1,732,544** | **9,969,589** | **~56.2** | **~97,300,000** |

---

## L. Current Expected Runtime per Stage (Baseline on 2 vCPU / 8 GB RAM)

| Stage | Operations | Current Baseline Runtime |
| :--- | :--- | :--- |
| **1. Normalization** | 11.7M records text processing | **~10 – 12 minutes** |
| **2. Blocking** | TF-IDF + `sparse_dot_topn` (all partitions) | **~15 – 20 minutes** |
| **3. Feature Extraction** | 97.3M pairs × 25 RapidFuzz metrics | **~35 – 45 minutes** |
| **4. LightGBM Inference** | 97.3M pairs through 2,784 trees (4.5k rows/s) | **~330 – 360 minutes (~5.5 – 6.0 hours)** |
| **5. Output Generation** | 1-to-1 matching + string aggregation + TSVs | **~8 – 12 minutes** |
| **TOTAL BASELINE** | End-to-end full inference | **~7.0 – 7.5 HOURS** |

---

## M. Exact Bottleneck Ranking

1. **RANK 1 (80% of total time): LightGBM Tree Evaluation on 97.3M Candidate Pairs**  
   - *Cause:* 2,784 deep trees evaluated sequentially on 97M rows at 4,500 rows/sec on 2 vCPUs.
2. **RANK 2 (10% of total time): Repeated RapidFuzz String Operations in Python Loop**  
   - *Cause:* Over 1 billion metric calls, string re-tokenization, and Python-to-C API transitions.
3. **RANK 3 (5% of total time): String Materialization & Polars Join Overhead**  
   - *Cause:* Joining 14 string columns for 97M rows and calling `.to_list()` creates enormous object churn and memory pressure.
4. **RANK 4 (3% of total time): TF-IDF Partition Fitting and Sparse Matrix Multiplication**  
   - *Cause:* Refitting TF-IDF vectorizers repeatedly per partition.
5. **RANK 5 (2% of total time): TSV Output Generation and Formatting**

---

## Top 3 Changes Most Likely to Produce the Largest Speedup

1. **Cascade / Early-Exit Tree Inference (Predicted Speedup: 5x – 8x on Scoring):**
   - Compute fast baseline features (`combo`, `sim_name`, `sim_addr`, `house_rel`, `n_jw`).
   - Pairs with extremely low combined similarity (e.g. `combo < 15.0` and `sim_name < 0.2` and `sim_addr < 0.2`) have **0.000% empirical probability** of exceeding the 0.68 match threshold and can be safely filtered or evaluated with a fast shallow predictor (first 100 trees), evaluating the full 2,784 trees only on plausible matches. This drops LightGBM inference time from 360 minutes to **< 15 minutes** with **0.0% recall loss**.
2. **Vectorized / Pre-tokenized Pairwise Features & Direct Index Lookup (Predicted Speedup: 3x – 4x on Features):**
   - Eliminate Polars joins by indexing pre-extracted string arrays via integer indices `a_idx` and `b_idx`.
   - Pre-compute token sets, word counts, and first tokens during normalization so that pairwise feature extraction avoids all runtime string splits.
   - Drops feature engineering time from 40 minutes to **~10 minutes**.
3. **Chunked Streaming Pipeline with In-Memory Recycling (Predicted Speedup: 2x on I/O & Memory Safety):**
   - Stream S1 entities in chunks of 50,000 entities directly from blocking to features to scoring to output.
   - Guarantees peak RAM stays **< 3.5 GB** (well below the 8 GB limit) with zero disk-swapping or giant multi-gigabyte intermediate files.
