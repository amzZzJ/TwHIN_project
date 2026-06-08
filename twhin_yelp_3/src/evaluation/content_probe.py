
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.common import load_model


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)




def load_reviews_raw(data_dir: Path, state: str, biz_ids: set[str]) -> pd.DataFrame:
    rows = []
    path = data_dir / "yelp_academic_dataset_review.json"

    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)

            if r["business_id"] not in biz_ids:
                continue

            rows.append(
                {
                    "user_id": r["user_id"],
                    "business_id": r["business_id"],
                    "date": r["date"],
                    "useful": int(r["useful"]),
                    "stars": float(r["stars"]),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["out_dir"])
    data_dir = Path(cfg["data_dir"])
    state = cfg["state"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)

    emb = model.entity_emb.weight.detach().cpu().numpy().astype(np.float32)

    entities = pd.read_parquet(out_dir / "entities.parquet")

    users = entities[entities.entity_type == "user"][["original_id", "entity_id"]]
    businesses = entities[entities.entity_type == "business"][["original_id", "entity_id"]]

    user_map = dict(zip(users.original_id, users.entity_id))
    biz_map = dict(zip(businesses.original_id, businesses.entity_id))

    biz_ids = set(biz_map.keys())

    review_df = load_reviews_raw(data_dir, state, biz_ids)

    review_df["date"] = pd.to_datetime(review_df["date"])

    train_end = pd.Timestamp(cfg["train_end"])
    val_end = pd.Timestamp(cfg["val_end"])

    train_df = review_df[review_df["date"] <= train_end].copy()
    test_df = review_df[review_df["date"] > val_end].copy()

    # Binary label: useful > 0
    train_df["label"] = (train_df["useful"] > 0).astype(int)
    test_df["label"] = (test_df["useful"] > 0).astype(int)

    # map ids
    train_df["u"] = train_df["user_id"].map(user_map)
    train_df["b"] = train_df["business_id"].map(biz_map)
    test_df["u"] = test_df["user_id"].map(user_map)
    test_df["b"] = test_df["business_id"].map(biz_map)

    train_df = train_df.dropna(subset=["u", "b"])
    test_df = test_df.dropna(subset=["u", "b"])

    def make_features(df: pd.DataFrame) -> np.ndarray:
        u = emb[df["u"].astype(int).to_numpy()]
        b = emb[df["b"].astype(int).to_numpy()]

        return np.concatenate([u, b], axis=1)

    X_train = make_features(train_df)
    y_train = train_df["label"].to_numpy()

    X_test = make_features(test_df)
    y_test = test_df["label"].to_numpy()

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, prob)
    roc = roc_auc_score(y_test, prob)
    prevalence = float(np.mean(y_test))

    print("Useful/content probe (label: useful>0)")
    print(f"  test size:   {len(y_test)}")
    print(f"  prevalence:  {prevalence:.4f}  (PR-AUC baseline)")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  ROC-AUC:     {roc:.4f}")


if __name__ == "__main__":
    main()
