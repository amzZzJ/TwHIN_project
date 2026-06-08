
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from recsys.TwHIN_project.twhin_yelp_3.src.data.dataset import load_relation_map
from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.common import entity_embeddings, load_model, relation_vector


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_personalized(cfg: dict, model, verbose: bool = True) -> dict:
    """Search-ranking analog (paper Sec 6.2, Table 3).

    A "session" is a (user, category) pair. Within the category the model ranks
    candidate businesses; the user's test-time reviewed businesses in that
    category are the positives. We score a candidate business b with the paper's
    triple score f(user, review, b) = (theta_u + theta_review)^T theta_b, then
    report MAP and averaged ROC-AUC across sessions (per-session, then averaged).
    """
    out_dir = Path(cfg["out_dir"])
    pcfg = cfg.get("personalized", {})
    min_session = int(pcfg.get("min_session", 3))
    max_users = int(pcfg.get("max_users", 4000))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    emb = entity_embeddings(model)
    rel_map = load_relation_map(out_dir)
    r_review = relation_vector(model, rel_map["review"])

    entities = pq.read_table(out_dir / "entities.parquet").to_pandas()
    biz_ids = entities.loc[entities.entity_type == "business", "entity_id"].astype(int).to_numpy()

    # business -> categories  (from in_category edges)
    cat_path = out_dir / "in_category_train.tsv"
    if not cat_path.exists():
        if verbose:
            print("[personalized] in_category edges missing; skipped")
        return {}
    cat_df = pd.read_csv(cat_path, sep="\t", header=None, names=["src", "rel", "dst"])
    cat_to_biz: dict[int, list[int]] = {}
    for b, c in zip(cat_df["src"].astype(int), cat_df["dst"].astype(int)):
        cat_to_biz.setdefault(c, []).append(b)

    # user test engagements
    test_path = out_dir / "review_test.tsv"
    train_path = out_dir / "review_train.tsv"
    if not test_path.exists():
        if verbose:
            print("[personalized] review_test missing; skipped")
        return {}
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["src", "rel", "dst"])
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["src", "rel", "dst"]) if train_path.exists() else None

    # which categories each business belongs to
    biz_cats: dict[int, list[int]] = {}
    for c, blist in cat_to_biz.items():
        for b in blist:
            biz_cats.setdefault(b, []).append(c)

    seen_train: dict[int, set] = {}
    if train_df is not None:
        for u, b in zip(train_df["src"].astype(int), train_df["dst"].astype(int)):
            seen_train.setdefault(u, set()).add(b)

    # build (user, category) -> positive businesses
    sessions: dict[tuple[int, int], set] = {}
    for u, b in zip(test_df["src"].astype(int), test_df["dst"].astype(int)):
        for c in biz_cats.get(b, []):
            sessions.setdefault((u, c), set()).add(b)

    users_pool = list({u for (u, _) in sessions})
    if len(users_pool) > max_users:
        keep = set(rng.choice(users_pool, size=max_users, replace=False).tolist())
        sessions = {k: v for k, v in sessions.items() if k[0] in keep}

    aps, rocs = [], []
    n_eval = 0
    for (u, c), positives in sessions.items():
        cand = cat_to_biz.get(c, [])
        # exclude businesses already engaged in training (no leakage / trivial)
        train_seen = seen_train.get(u, set())
        cand = [b for b in cand if b not in train_seen or b in positives]
        cand = list(dict.fromkeys(cand))  # dedup, keep order
        pos = positives & set(cand)
        neg = [b for b in cand if b not in pos]
        if len(pos) < 1 or len(neg) < 1 or len(cand) < min_session:
            continue

        q = emb[u] + r_review  # (theta_u + theta_review)
        cand_arr = np.asarray(cand)
        scores = emb[cand_arr] @ q  # dot product == paper f(.)
        labels = np.array([1 if b in pos else 0 for b in cand], dtype=int)

        try:
            rocs.append(roc_auc_score(labels, scores))
        except ValueError:
            continue
        aps.append(average_precision_score(labels, scores))
        n_eval += 1

    res = {
        "sessions": int(n_eval),
        "MAP": float(np.mean(aps)) if aps else 0.0,
        "avg_ROC": float(np.mean(rocs)) if rocs else 0.0,
    }
    if verbose:
        print("Personalized ranking (per user-category session)  --  Table 3 analog")
        print(f"  sessions evaluated: {res['sessions']}")
        print(f"  MAP:      {res['MAP']:.4f}")
        print(f"  avg ROC:  {res['avg_ROC']:.4f}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    res = run_personalized(cfg, model)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
