# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 25 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 business records against the deduplicated Source 1 reference with a
classic, fully reproducible **blocking + pairwise classifier** pipeline: language-agnostic
normalisation (including an Indic-script → Latin word dictionary learnt from the training ground
truth), two-channel TF-IDF candidate generation inside (country, state) partitions with forward *and*
reverse retrieval (99.2 % candidate recall at 56 candidates per entity), a 49-feature LightGBM
matcher, and an F₀.₅-optimised decision rule with one-to-one assignment. On a held-out validation
split of the training data the pipeline reaches **macro F₀.₅ = 0.9747** (blocking ceiling 0.9974),
runs end-to-end on a 16 GB laptop in a few hours (training ≈ 45 min, test inference ≈ 3 h), and uses no external data or services.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the 2.2 M reference entities and 10.3 M Source 2/3 training records showed:

- **Match structure.** Only 5.6 % of Source 1 entities are singletons; the median entity has 3
  matches (max 11), split roughly evenly between Source 2 and Source 3. Every matched Source 2/3
  record belongs to exactly one Source 1 entity (7.64 M matched ids, all unique), so the problem is a
  one-to-many assignment, not a free many-to-many link prediction. 26 % of Source 2/3 records are
  distractors that match nothing.
- **Names are not identifying on their own.** 690 K of the 2.2 M reference names are exact
  duplicates of another reference name (e.g. 192 different "Mumbai India …" entities in
  Maharashtra), and 139 K distractor records share an exact normalised name with some Source 1 entity
  at a different address. The address, and in particular the house number, is what disambiguates.
- **Name noise** (synthetic, heavy): case, accented vowels ("Émpire"), 1/l and 0/o substitutions,
  transposed and dropped letters, legal suffix moved / dropped / bracketed ("((LLC))", "[Group]"),
  word-order swaps, generic words injected ("Center", "Partners", "Services"), *former-name* markers
  ("f/k/a", "née"), truncation to the first word(s), and **domain-style names**
  ("modernconstructionenterprises.com", 3.4 % of records). 7.3 % of Source 2/3 names are written in an
  **Indic script** (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Punjabi, Odia)
  as a phonetic transliteration of the English name. Some matches carry an entirely different DBA
  name at the same address.
- **Address noise:** abbreviations (Rd/Road, St/Street/Saint), ordinal typos ("102RD"), component
  reordering, missing components (no house number, no state, 3.3 % fully missing), null tokens
  ("<NULL>", "N/A"), unit / PMB / PO-box lines added or removed, state written as code, full name or in
  a regional script ("महाराष्ट्र", "ಕರ್ನಾಟಕ"), "Calcutta"/"Kolkata"-style aliases, and small house-number
  perturbations (924 → 922). Telangana records are labelled Andhra Pradesh in 0.5 % of true pairs.
- **Unseen country.** The test set adds France (259 K Source 1 entities). French records use their own
  abbreviations (R./Rue, AV/Avenue, Bd/Boulevard), legal forms (SARL, SAS, SASU, SCI, EURL) and a
  region / department / city hierarchy (Hauts-de-France ↔ Nord ↔ Lille). Every learned component
  must therefore be language-agnostic.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier (LightGBM) + global one-to-one assignment  
**Core Innovation:**

1. A **word-level transliteration dictionary learnt from the training ground truth**: matched
   (Latin, Indic) name pairs are aligned token by token; the resulting 1,347-entry table
   (e.g. प्राइवेट → private, కన్‌స్ట్రక్షన్ → construction) covers 96 % of the Indic tokens in the test set
   and lets Indic-script names be matched with ordinary string similarity.
2. **Bidirectional two-channel blocking** inside (country, state) partitions — name char-3-grams and
   address word tokens, each queried from the Source 1 side (top-k per entity) *and* from the
   Source 2/3 side (top-3 per record). The reverse queries are what lift recall from 97.0 % to
   99.2 %: they guarantee that every candidate record reaches its best reference entities even when
   dozens of same-name entities crowd the forward top-k.
3. **Context and competition features** for the classifier: how many reference entities share this
   name / address in the country, the candidate's rank and gap to the best candidate of the entity,
   and how strongly *other* reference entities claim the same candidate — followed by a hard
   one-to-one assignment at decision time.

---

## 3. Candidate Generation (Blocking)

**Normalisation first.** Every record is mapped to `name_norm`, `name_core` (legal-form and stop
tokens removed), `name_compact` (spaces removed, so "modernconstructionenterprises.com" equals
"Modern Construction Enterprises"), `addr_norm` (accent-folded, abbreviation-canonicalised, filler
tokens dropped, ordinals → digits, leading zeros stripped), `addr_nums` (digit-bearing tokens),
`house_no` (first number of the first street-like component) and `state` (resolved from 2-letter
codes, full names, regional-script spellings and — for France — regions, departments and cities).
State resolution is two-pass so that "Washington, DC" keeps the city token and resolves to DC.

- **Blocking keys used:**
  - *Partition key:* `country` × `state` (Telangana merged with Andhra Pradesh, DC with WA).
    Source 2/3 records with no resolvable state (≈3 %) are appended to every partition of their
    country; Source 1 records with no state are compared against the whole country.
  - *Name channel:* TF-IDF (sublinear tf) over character 3-grams of `name_compact`, cosine top-20 per
    Source 1 record (threshold 0.25) and top-3 Source 1 records per candidate record.
  - *Address channel:* TF-IDF over word unigrams of `addr_norm + addr_nums`, cosine top-15 per
    Source 1 record (threshold 0.20) and top-3 per candidate record.
  - Retrieval uses `sparse_dot_topn` (multithreaded sparse top-n matrix products); the whole training
    sample (431 K entities × 10.3 M records) blocks in ≈4 minutes on a laptop.
- **Candidate pairs generated:** 24.3 M for the 431 K-entity training sample (56.3 per entity);
  96.7 M for the 1.73 M test entities (55.8 per entity).
- **How you ensured true matches were not lost:** recall was measured against the ground truth on a
  training sample after every change (single forward channel: 97.0 %; + state merging and reverse
  channels: **99.24 %**, 97.5 % of entities with *all* matches recovered). We deliberately did not
  cap the candidate list: capping to the 30 best-scoring candidates per entity costs 1.5 recall
  points because reverse-channel candidates have low TF-IDF cosine but are frequently true.
  The remaining 0.8 % of misses are records whose address is missing and whose name is both
  corrupted and generic, plus a few pairs where both name and house number were changed.

---

## 4. Matching Model

**Features used (49):**

- Name features: rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` on
  `name_core`; `ratio` and Jaro-Winkler on `name_compact`; `ratio` on the full name (legal form
  included); token Jaccard; first-token agreement; prefix relation; exact compact equality; token
  counts; blocking TF-IDF cosine; candidate-side flags for non-Latin script and domain-style names.
- Address features: `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and token
  Jaccard on `addr_norm`; Jaccard and intersection size of the digit tokens; house-number relation
  (equal / one missing / conflict) and house-number edit ratio; state relation (equal / one missing /
  conflict); token counts; address-missing flag; blocking TF-IDF cosine.
- Context features: number of reference entities in the country sharing the same `name_core`
  (`amb_name`) and the same normalised address (`amb_addr`); per-entity candidate count, max /
  rank / gap of name, address and combined similarity, number of near-perfect name and address
  candidates; per-candidate degree, best combined similarity offered by any competing reference
  entity, rank and gap of this entity among them.
- No country / language specific feature is used, so the model transfers to France unchanged.

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.05, 2,784 rounds chosen by
early stopping, feature / bagging fraction 0.8). Trained on 7.6 M labelled pairs (all 1.19 M
positives, 35 % of negatives) from the 431 K-entity training sample; the sample consists of complete
(country, state) partitions (8 US states, 6 Indian states) so that rank and competition features
are computed exactly as at test time. The most important features by gain are the combined
name+address similarity, its gap to the entity's best candidate, the competition gap, the address
TF-IDF cosine and the digit-token Jaccard.

**Threshold selection method:** macro F₀.₅ (singletons included) was swept on the 20 % validation
split of the sample, with and without one-to-one assignment. The optimum is **p ≥ 0.68 with
one-to-one assignment** (each Source 2/3 record is given to the reference entity with the highest
probability, ties to the first). The curve is flat between 0.60 and 0.75 (F₀.₅ ≥ 0.974), so the
operating point is robust.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9747** on 86,155 held-out training entities (India 0.9722,
  US 0.9773). Blocking ceiling (oracle classification of the candidate set): 0.9974. Per-entity
  scores by number of true matches: singletons 0.964 (177 of 4,852 singletons wrongly given a match),
  one match 0.932, two 0.970, three 0.976, four or more ≥ 0.98.
- **Common false positives (wrong merges):** 3,097 pairs. 88 % are distractor records that match
  nothing in the reference set and were generated as near-copies of a reference entity: identical
  name and street, house number changed (2246/b → 2250/b) or one word of the name swapped
  ("Wayanad Garments" → "Wayanad Sky"); these are indistinguishable from true matches that carry the
  same house-number perturbation. The rest are generic one-word Indian names ("SS Agro") whose
  candidate shares only the city.
- **Common false negatives (missed matches):** 14,167 pairs, of which 2,241 were never candidates
  and 11,912 scored below the threshold. 41 % of the missed records have **no address** and a
  generic or truncated name ("Traders", "Big Seafood"), where the prior probability of a wrong merge
  is high and the F₀.₅ objective favours abstaining; most of the remainder combine a house-number
  change with a name edit. Only 14 pairs were lost to the one-to-one constraint.

---

## 6. Conclusion

A carefully normalised, bidirectional TF-IDF blocking stage plus a gradient-boosted matcher with
ambiguity- and competition-aware features resolves this benchmark at macro F₀.₅ ≈ 0.975 on held-out
data while staying language-agnostic enough for an unseen country. The two lessons we take away
are that (i) recall is won at blocking time by querying from *both* sides of the match and by
partitioning on a robustly resolved state, and (ii) precision under F₀.₅ comes from modelling how
ambiguous a name or address is within the reference set rather than from raw string similarity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md` for exact commands):

| file | role |
| --- | --- |
| `src/translit.py` | learns the Indic-script → Latin word dictionary from `train_ground_truth.tsv` |
| `src/normalize.py` | name / address canonicalisation, state resolution, house-number extraction (multiprocess) |
| `src/block.py` | partitioned two-channel forward + reverse TF-IDF retrieval; recall evaluation |
| `src/features.py` | 49 pairwise / context features, computed per partition chunk with bounded memory |
| `src/train.py` | LightGBM training, macro-F₀.₅ threshold sweep, per-country report |
| `src/scoring.py` | challenge metric and decision rule (threshold + one-to-one) |
| `src/predict.py` | end-to-end test inference producing `output/matching_results.tsv` and `output/candidate_pairs.tsv` |

Entry point: `predict.py` (after `translit.py`, `normalize.py`, `block.py`, `features.py`, `train.py`
have produced `work/model/`). Hardware used: Apple M4 laptop, 10 cores, 16 GB RAM, no GPU; total
wall-clock ≈ 45 min for training and ≈ 180 min for test inference.

### B. Additional Results

Validation threshold sweep (one-to-one assignment on):

| threshold | 0.50 | 0.55 | 0.60 | 0.65 | 0.68 | 0.70 | 0.75 | 0.80 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro F₀.₅ | 0.9717 | 0.9729 | 0.9740 | 0.9746 | **0.9747** | 0.9747 | 0.9743 | 0.9735 | 0.9688 |

Blocking ablation on a 100 K-entity random training sample: forward channels only
(top-15 / top-15) 97.00 % pair recall, 27 candidates per entity; with state merging, top-20 names
and reverse top-3 channels: 99.24 %, 56 candidates per entity. Per channel alone: name 90.9 %,
address 90.7 %.

Transliteration dictionary: 551 K matched pairs with an Indic-script name, 511 K aligned
token-by-token, 1,347 entries kept (≥ 2 supporting pairs, ≥ 50 % agreement); token coverage on the
test set 96.3 %, 85.6 % of Indic test names fully transliterated.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
