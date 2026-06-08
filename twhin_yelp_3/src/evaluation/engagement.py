
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.common import entity_embeddings, load_model


def relative_cross_entropy(y_true: np.ndarray, prob: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(prob, eps, 1 - eps)
    ce_model = log_loss(y_true, p, labels=[0, 1])
    base = float(np.mean(y_true))
    base = min(max(base, eps), 1 - eps)
    ce_base = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return 100.0 * (ce_base - ce_model) / ce_base


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_features(emb: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    ue = emb[u]
    be = emb[b]
    return np.concatenate([ue, be, ue * be, np.abs(ue - be)], axis=1)


def sample_negatives(df: pd.DataFrame, biz_ids: np.ndarray, n_neg: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos_pairs = set(zip(df.src.astype(int), df.dst.astype(int)))
    users = df.src.astype(int).to_numpy()
    neg_u, neg_b = [], []
    while len(neg_u) < n_neg:
        u = int(rng.choice(users))
        b = int(rng.choice(biz_ids))
        if (u, b) not in pos_pairs:
            neg_u.append(u)
            neg_b.append(b)
    return pd.DataFrame({"src": neg_u, "dst": neg_b, "label": 0})


def run_engagement(cfg: dict, model, relation: str = "review", verbose: bool = True) -> dict:
    """Binary engagement prediction with TwHIN embeddings as frozen features.

    relation in {"review", "tip"} selects which engagement edge type is the
    positive class. Evaluating both lets us test the paper's claim that
    low-coverage relations (tip) benefit from joint training while high-coverage
    relations (review) do not lose performance.
    """
    out_dir = Path(cfg["out_dir"])
    emb = entity_embeddings(model)

    entities = pq.read_table(out_dir / "entities.parquet").to_pandas()
    biz_ids = entities.loc[entities.entity_type == "business", "entity_id"].astype(int).to_numpy()

    train_path = out_dir / f"{relation}_train.tsv"
    test_path = out_dir / f"{relation}_test.tsv"
    if not train_path.exists() or not test_path.exists():
        if verbose:
            print(f"[engagement] {relation} splits missing; skipped")
        return {}

    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["src", "rel", "dst"])
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["src", "rel", "dst"])
    if len(train_df) == 0 or len(test_df) == 0:
        if verbose:
            print(f"[engagement] {relation} empty; skipped")
        return {}

    train_pos = train_df[["src", "dst"]].copy(); train_pos["label"] = 1
    test_pos = test_df[["src", "dst"]].copy(); test_pos["label"] = 1
    train_neg = sample_negatives(train_pos, biz_ids, len(train_pos), seed=42)
    test_neg = sample_negatives(test_pos, biz_ids, len(test_pos), seed=43)

    train_all = pd.concat([train_pos, train_neg], ignore_index=True)
    test_all = pd.concat([test_pos, test_neg], ignore_index=True)

    X_train = build_features(emb, train_all.src.to_numpy(), train_all.dst.to_numpy())
    X_test = build_features(emb, test_all.src.to_numpy(), test_all.dst.to_numpy())
    y_train = train_all.label.to_numpy()
    y_test = test_all.label.to_numpy()

    clf = SGDClassifier(loss="log_loss", random_state=int(cfg.get("seed", 42)))
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]

    res = {
        "relation": relation,
        "test_size": int(len(y_test)),
        "prevalence": float(np.mean(y_test)),
        "ROC-AUC": float(roc_auc_score(y_test, prob)),
        "PR-AUC": float(average_precision_score(y_test, prob)),
        "RCE": float(relative_cross_entropy(y_test, prob)),
    }
    if verbose:
        print(f"Engagement ranking ({relation} edges, business negatives)")
        print(f"  test size:   {res['test_size']}")
        print(f"  prevalence:  {res['prevalence']:.4f}")
        print(f"  ROC-AUC:     {res['ROC-AUC']:.4f}")
        print(f"  PR-AUC:      {res['PR-AUC']:.4f}")
        print(f"  RCE:         {res['RCE']:.2f}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--relation", choices=["review", "tip"], default="review")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    run_engagement(cfg, model, relation=args.relation)


if __name__ == "__main__":
    main()
