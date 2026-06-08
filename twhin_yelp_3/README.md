
# TwHIN-style heterogeneous graph embeddings on Yelp (single-GPU)

Reproduction of the core TwHIN idea (El-Kishky et al., *TwHIN: Embedding the
Twitter Heterogeneous Information Network for Personalized Recommendation*,
KDD 2022) on the public Yelp Open Dataset, with a lightweight PyTorch pipeline
that fits a single GPU.

## What this reproduces (mapped to the paper)
- **Heterogeneous KGE with TransE** — paper Sec 4.1, Eq. 1: scoring is the
  dot-product translation `f(s,r,t) = (θ_s + θ_r)·θ_t` (NOT negative L2 distance).
- **Negative-sampling objective** — Eq. 2, corrupting **both head and target**.
- **Mixture-of-embeddings** — Sec 4.4, Eq. 3: k-means over non-user entities,
  users represented as a distribution over their top-m engaged clusters,
  multi-query ANN retrieval.
- **Four downstream tasks** (Sec 6):
  1. Candidate generation / Who-to-Follow (Table 1)  → friend recommendation
  2. Engagement ranking / ads (Table 2)              → review/tip engagement, RCE
  3. Personalized / search ranking (Table 3)         → per (user, category) sessions, MAP / ROC
  4. Content classification / offensive (Table 4)    → `useful` probe, PR-AUC
- **Relation ablations** — Table 2 analog: single-relation and leave-one-out,
  plus the low- vs high-coverage co-training test (Sec 4.3 / 6.2).

## Relations in the Yelp HIN
`review` (high-coverage engagement), `tip` (low-coverage engagement),
`friend` (social graph), `in_category`, `located_in`.

## Install
```bash
pip install torch pandas pyarrow scikit-learn pyyaml tqdm faiss-cpu
```

## Dataset layout
Point `data_dir` in `configs/base.yaml` at the Yelp JSON files:
`yelp_academic_dataset_{business,review,tip,user}.json`.

## Pipeline
```bash
# 1) preprocess + temporal/leak-free splits + ablation folders
python -m src.data.preprocess --config configs/base.yaml

# 2) train the TwHIN-style baseline (dot-product TransE, NS, head+tail corruption, Adagrad)
python -m src.training.train --config configs/base.yaml --ablation all --loss ns

#    optional: sampled softmax + LogQ
python -m src.training.train --config configs/base.yaml --ablation all --loss sampled_softmax --logq

# 3) evaluate
python -m src.evaluation.retrieval     --config configs/base.yaml --ckpt outputs/checkpoints/all__ns.pt   # Table 1: unimodal vs mixture
python -m src.evaluation.engagement    --config configs/base.yaml --ckpt outputs/checkpoints/all__ns.pt --relation review
python -m src.evaluation.engagement    --config configs/base.yaml --ckpt outputs/checkpoints/all__ns.pt --relation tip
python -m src.evaluation.personalized  --config configs/base.yaml --ckpt outputs/checkpoints/all__ns.pt   # Table 3 analog
python -m src.evaluation.content_probe --config configs/base.yaml --ckpt outputs/checkpoints/all__ns.pt   # Table 4 analog

# 4) full relation-ablation study (Table 2 analog + low/high-coverage test)
python -m src.training.run_ablations    --config configs/base.yaml --loss ns
python -m src.evaluation.ablation_report --config configs/base.yaml --loss ns
```

## Key config knobs (`configs/base.yaml`)
- `score_fn`: `dot` (paper, default) | `l2` (classic TransE, for comparison)
- `optimizer`: `adagrad` (paper, default) | `adamw`
- `negative_sampling.strategy`: `frequency` (power 0.75) | `uniform`
- `negative_sampling.corrupt`: `both` (paper, default) | `tail` | `head`
- `mixture`: `n_clusters`, `top_m`, `kmeans_iters`
- `personalized`: `min_session`, `max_users`

## Research questions
1. Does the heterogeneity gain reproduce on open Yelp data?
2. Which relations help most (social `friend` vs low-coverage `tip`)?
3. Do low-coverage relations benefit from joint training with high-coverage
   ones, while the reverse does not? (per-relation ablation report)
4. Mixture-of-embeddings vs single (unimodal) embedding for candidate generation.
5. Sampled softmax + LogQ vs classic negative sampling.

## Ablation folders produced by preprocessing
`ablation_{all,no_tip,no_friend,no_review,review_only,tip_only,friend_only,review_friend,review_tip}`
