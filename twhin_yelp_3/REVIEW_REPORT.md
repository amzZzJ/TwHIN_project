# Review: agreement with the TwHIN paper, fixes, and completed work

This report reviews the existing Yelp reproduction of **TwHIN** (El-Kishky et al.,
*Embedding the Twitter Heterogeneous Information Network for Personalized
Recommendation*, KDD 2022) against (a) the source paper and (b) the project
proposal (`main.pdf`). It records the discrepancies found, the fixes applied,
the missing components that were built, and how the code was validated.

> **Scope note.** The real Yelp Open Dataset and a GPU were not available in this
> environment. The pipeline was therefore validated for **correctness** on a
> synthetic dataset generated in the exact Yelp JSON schema. The numbers below
> are code sanity-checks, **not** a reproduction of the paper's absolute values —
> those require running on the real Yelp files referenced in `configs/base.yaml`.

---

## 1. Discrepancies found vs. the source paper / proposal

| # | Where | Paper / proposal says | Original code did | Severity |
|---|-------|----------------------|-------------------|----------|
| 1 | `models/transe.py` | Eq. 1: dot-product translation `f(s,r,t) = (θ_s+θ_r)·θ_t`; proposal restates the same formula | Negative L2 distance `−‖θ_s+θ_r−θ_t‖` | **High** — wrong scoring geometry, different objective surface |
| 2 | `training/train.py`, `data/sampler.py` | Eq. 2 / proposal: corrupt **source and target**, `N = {(s,r,t′)} ∪ {(s′,r,t)}` | Corrupted the **tail only** | **Medium** — biased negatives, head embeddings under-trained |
| 3 | `training/train.py` | Sec 4.1: optimize with **Adagrad** | AdamW | **Low** — different optimizer dynamics |

All three were genuine deviations from what the proposal explicitly committed to
reproduce. None of them were documented as intentional simplifications.

## 2. Work that was specified but **not done**

The proposal commits to four downstream tasks and a set of ablations. Status of
the original code:

| Component | Proposal ref | Original status |
|-----------|-------------|-----------------|
| Candidate generation (Who-to-Follow, Table 1) | Sec 6.1 | Done (unimodal only) |
| Engagement ranking (ads, Table 2), RCE | Sec 6.2 | Done (review only) |
| **Personalized / search ranking (Table 3), MAP + avg ROC** | Sec 6.2 | **Missing** |
| Content classification (offensive, Table 4), PR-AUC | Sec 6.2 | Done |
| **Mixture-of-embeddings (Sec 4.4, Eq. 3)** + unimodal-vs-mixture (Table 1) | Sec 4.4 | **Missing** |
| **Relation ablation driver + Table 2 aggregation** | proposal "Ablation по типам отношений" | **Missing** (folders built, nothing ran/aggregated them) |
| **Per-relation eval / low-vs-high-coverage test** | proposal "Per-relation evaluation" | **Missing** |
| Sanity baselines (random / most-popular) | proposal "Sanity-бейзлайны" | Done |

## 3. Fixes applied

1. **TransE scoring → dot product (Eq. 1).** `score_fn="dot"` is the default;
   `score_fn="l2"` retained for comparison. Stored in the checkpoint so eval
   reconstructs the right model.
2. **Head + tail corruption (Eq. 2).** Added `TransE.head_scores`; negative
   sampling now splits `negative_samples` between head- and tail-corruptions
   (`negative_sampling.corrupt: both|tail|head`, default `both`).
3. **Adagrad optimizer (Sec 4.1).** `optimizer: adagrad|adamw`, default `adagrad`.

## 4. Missing components built

- **Mixture-of-embeddings** (`evaluation/retrieval.py`): MiniBatchKMeans over
  non-user (business) entities; per-user cluster-engagement distribution
  `P(c|u) ∝ count(u,c)` over the top-m clusters (Eq. 3); multi-query ANN
  retrieval with per-cluster budget proportional to mixture weight. The
  retrieval evaluator now prints a **Table 1 analog**: unimodal vs mixture vs
  random vs most-popular, with the mixture/unimodal lift.
- **Personalized ranking** (`evaluation/personalized.py`): a "session" is a
  `(user, category)` pair; businesses in the category are ranked by the paper's
  triple score `f(user, review, business)`; reports **MAP** and **averaged
  ROC-AUC** across sessions (Table 3 analog). Training-time engagements are
  excluded from candidates to avoid leakage.
- **Per-relation engagement** (`evaluation/engagement.py`): `--relation
  review|tip`, refactored into an importable `run_engagement(...)`.
- **Ablation study** (`training/run_ablations.py`, `evaluation/ablation_report.py`):
  trains the single-relation, leave-one-out, pairwise, and full-graph models,
  then aggregates a **Table 2 analog** and prints the **low- vs high-coverage
  co-training comparison** — does `tip` (low-coverage) gain from joint training
  while `review` (high-coverage) does not?
- Added the `no_review` leave-one-out ablation so the leave-one-out set is
  complete for the three core relations.
- Corrected the notebooks' "comparison to paper" table (was vague placeholders
  like `~0.10-0.30`) to the paper's actual Table 1/2/3/4 values.

## 5. Validation (synthetic Yelp-schema data)

A 1,500-user / 300-business synthetic graph with category-driven engagement and
interest-clustered friendships was generated to exercise every code path. All of
the following ran clean end-to-end:

- preprocess → temporal split → 9 ablation folders (leak-free friend split)
- training: NS (dot, both-corruption, Adagrad) **and** sampled-softmax + LogQ
- all four downstream evaluators
- 6-ablation study + aggregation report, including the low/high-coverage test
- both `uniform` and `frequency` sampling; `dot` and `l2` scoring

Example report output (synthetic — illustrative only):

```
Relation ablation  --  Table 2 analog
        ablation | review_RCE | tip_RCE | search_MAP | ...
             all |    -0.18   |  -7.09  |   0.083    |
     review_only |    -1.11   |  -4.15  |   0.084    |
        tip_only |    -1.07   | -11.47  |   0.073    |

Low- vs high-coverage co-training:
  tip (low-coverage)  RCE: solo=-11.47  joint(all)=-7.09  delta=+4.38  (low gains from joint)
  review (high-cov.)  RCE: solo=-1.11   joint(all)=-0.18  delta=+0.93
```

**Honest caveats about the synthetic numbers:**
- Absolute metrics are weak/near-baseline (RCE can be negative) because the
  synthetic labels carry little real signal — this is a *code* check, not a
  result.
- The unimodal-vs-mixture lift swings run-to-run on synthetic data because the
  synthetic users have only two latent interests, so there is little
  multi-interest structure for the mixture to exploit. The paper's large
  Table 1 gain (0.58% → 3.70% R@10) depends on real multi-interest users; the
  mechanism is implemented faithfully, but its magnitude must be measured on the
  real Yelp graph.

## 6. Paper reference values (for comparison once run on real Yelp)

- **Table 1** (Who-to-Follow): unimodal R@10/20/50 = 0.58 / 1.02 / 2.06 %;
  mixture = 3.70 / 5.53 / 8.79 % (>300 % R@10 lift).
- **Table 2** (ads RCE): baselines ≈ 13–21; adding U/A/T entity embeddings
  increases RCE; online +2.38 RCE, −10.3 % cost-per-conversion.
- **Table 3** (search): MAP 55.7 → 57.0, avg ROC 57.9 → 59.6 (−2.8 % / −4.0 %
  relative error). Author embeddings help only combined with user embeddings.
- **Table 4** (offensive): PR-AUC RoBERTa 0.4123, BERTweet 0.4692, +TwHIN-Author
  0.5161 (+9.09 % rel on Collection1; neutral on Collection2).

## 7. How to reproduce on real data

See `README.md`. In short: point `configs/base.yaml:data_dir` at the Yelp JSON
files, then run preprocess → `run_ablations` → the five evaluators /
`ablation_report`.
