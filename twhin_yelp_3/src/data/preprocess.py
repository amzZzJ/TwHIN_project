
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .temporal_split import temporal_split


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_json_lines(path: Path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    state = cfg["state"]
    include_categories = bool(cfg.get("include_categories", True))
    include_city = bool(cfg.get("include_city", True))
    max_categories = int(cfg.get("max_categories_per_business", 5))

    # ------------------------------------------------------------------
    # Businesses
    # ------------------------------------------------------------------
    print("Loading businesses...")
    businesses: List[dict] = []
    for b in read_json_lines(data_dir / "yelp_academic_dataset_business.json"):
        businesses.append(
            {
                "business_id": b["business_id"],
                "state": b["state"],
                "city": b["city"],
                "categories": b["categories"],
            }
        )

    biz_df = pd.DataFrame(businesses)
    biz_df = biz_df[biz_df["state"] == state].copy()
    biz_ids = set(biz_df["business_id"])
    print(f"Businesses in {state}: {len(biz_df)}")

    # ------------------------------------------------------------------
    # Reviews (high coverage engagement)
    # ------------------------------------------------------------------
    print("Loading reviews...")
    reviews = []
    for r in read_json_lines(data_dir / "yelp_academic_dataset_review.json"):
        if r["business_id"] in biz_ids:
            reviews.append(
                {
                    "user_id": r["user_id"],
                    "business_id": r["business_id"],
                    "date": r["date"],
                    "stars": r["stars"],
                    "useful": r["useful"],
                }
            )
    review_df = pd.DataFrame(reviews)
    print(f"Reviews: {len(review_df):,}")

    # ------------------------------------------------------------------
    # Tips (low coverage engagement)
    # ------------------------------------------------------------------
    print("Loading tips...")
    tips = []
    for t in read_json_lines(data_dir / "yelp_academic_dataset_tip.json"):
        if t["business_id"] in biz_ids:
            tips.append(
                {
                    "user_id": t["user_id"],
                    "business_id": t["business_id"],
                    "date": t["date"],
                }
            )
    tip_df = pd.DataFrame(tips)
    print(f"Tips: {len(tip_df):,}")

    # ------------------------------------------------------------------
    # Users + friends
    # ------------------------------------------------------------------
    pa_user_ids = set(review_df["user_id"]) | set(tip_df["user_id"])

    print("Loading users + friends...")
    users = []
    friend_edges = []

    for u in read_json_lines(data_dir / "yelp_academic_dataset_user.json"):
        if u["user_id"] not in pa_user_ids:
            continue

        users.append(
            {
                "user_id": u["user_id"],
                "review_count": u["review_count"],
                "fans": u["fans"],
                "average_stars": u["average_stars"],
            }
        )

        if u["friends"] and u["friends"] != "None":
            for friend_id in u["friends"].split(", "):
                friend_id = friend_id.strip()
                if friend_id in pa_user_ids:
                    friend_edges.append({"src": u["user_id"], "dst": friend_id})

    user_df = pd.DataFrame(users)
    friends_df = pd.DataFrame(friend_edges)

    if len(friends_df) > 0:
        friends_df["key"] = friends_df.apply(lambda r: tuple(sorted([r.src, r.dst])), axis=1)
        friends_df = friends_df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)

    print(f"Users loaded: {len(user_df):,}")
    print(f"Friend edges: {len(friends_df):,}")

    # ------------------------------------------------------------------
    # Temporal split for review/tip
    # ------------------------------------------------------------------
    train_end = cfg["train_end"]
    val_end = cfg["val_end"]

    review_train, review_val, review_test = temporal_split(review_df, train_end, val_end)
    tip_train, tip_val, tip_test = temporal_split(tip_df, train_end, val_end)

    # ------------------------------------------------------------------
    # Entity vocabulary
    # ------------------------------------------------------------------
    all_users = sorted(set(review_df["user_id"]) | set(tip_df["user_id"]))
    all_businesses = sorted(biz_df["business_id"])

    category_names = set()
    if include_categories:
        for cats in biz_df["categories"].fillna(""):
            if not cats:
                continue
            for c in cats.split(","):
                c = c.strip()
                if c:
                    category_names.add(c)

    city_names = set()
    if include_city:
        city_names = set(biz_df["city"].dropna().astype(str))

    user2id = {u: i for i, u in enumerate(all_users)}
    biz_offset = len(user2id)
    biz2id = {b: biz_offset + i for i, b in enumerate(all_businesses)}

    cat_offset = biz_offset + len(biz2id)
    cat2id = {c: cat_offset + i for i, c in enumerate(sorted(category_names))}

    city_offset = cat_offset + len(cat2id)
    city2id = {c: city_offset + i for i, c in enumerate(sorted(city_names))}

    total_entities = len(user2id) + len(biz2id) + len(cat2id) + len(city2id)

    print("Entity counts")
    print(f"  users:      {len(user2id):,}")
    print(f"  businesses: {len(biz2id):,}")
    print(f"  categories: {len(cat2id):,}")
    print(f"  cities:     {len(city2id):,}")
    print(f"  total:      {total_entities:,}")

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------
    rel2id: Dict[str, int] = {
        "review": 0,
        "tip": 1,
        "friend": 2,
    }

    if include_categories:
        rel2id["in_category"] = len(rel2id)

    if include_city:
        rel2id["located_in"] = len(rel2id)

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------
    def build_review_edges(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "src": df["user_id"].map(user2id),
                "dst": df["business_id"].map(biz2id),
                "relation": rel2id["review"],
            }
        ).dropna().astype({"src": int, "dst": int, "relation": int})

    def build_tip_edges(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "src": df["user_id"].map(user2id),
                "dst": df["business_id"].map(biz2id),
                "relation": rel2id["tip"],
            }
        ).dropna().astype({"src": int, "dst": int, "relation": int})

    friend_edges_all = pd.DataFrame(
        {
            "src": friends_df["src"].map(user2id),
            "dst": friends_df["dst"].map(user2id),
            "relation": rel2id["friend"],
        }
    ).dropna().astype({"src": int, "dst": int, "relation": int})

    friend_frac_test = float(cfg.get("friend_test_frac", 0.1))
    rng_friend = np.random.default_rng(int(cfg.get("seed", 42)))
    perm = rng_friend.permutation(len(friend_edges_all))
    n_test = int(len(friend_edges_all) * friend_frac_test)
    test_idx = set(perm[:n_test].tolist())

    is_test = friend_edges_all.index.isin(test_idx)
    friend_edges_mapped = friend_edges_all[~is_test].reset_index(drop=True)
    friend_test_raw = friend_edges_all[is_test].reset_index(drop=True)

    train_users = set(friend_edges_mapped["src"]) | set(friend_edges_mapped["dst"])
    friend_test_e = friend_test_raw[
        friend_test_raw["src"].isin(train_users) & friend_test_raw["dst"].isin(train_users)
    ].reset_index(drop=True)

    # category edges
    category_edges = []
    if include_categories:
        for row in biz_df.itertuples(index=False):
            cats = row.categories or ""
            if not cats:
                continue
            seen = set()
            for c in cats.split(","):
                c = c.strip()
                if not c or c not in cat2id or c in seen:
                    continue
                seen.add(c)
                category_edges.append(
                    {
                        "src": biz2id[row.business_id],
                        "dst": cat2id[c],
                        "relation": rel2id["in_category"],
                    }
                )
                if len(seen) >= max_categories:
                    break
    category_edges = pd.DataFrame(category_edges)

    # city edges
    city_edges = []
    if include_city:
        for row in biz_df.itertuples(index=False):
            city = str(row.city)
            if city in city2id:
                city_edges.append(
                    {
                        "src": biz2id[row.business_id],
                        "dst": city2id[city],
                        "relation": rel2id["located_in"],
                    }
                )
    city_edges = pd.DataFrame(city_edges)

    # temporal interaction edges
    review_train_e = build_review_edges(review_train)
    review_val_e = build_review_edges(review_val)
    review_test_e = build_review_edges(review_test)

    tip_train_e = build_tip_edges(tip_train)
    tip_val_e = build_tip_edges(tip_val)
    tip_test_e = build_tip_edges(tip_test)

    # ------------------------------------------------------------------
    # Train graph includes static friend/category/city edges
    # ------------------------------------------------------------------
    static_train_edges = [friend_edges_mapped]
    if len(category_edges) > 0:
        static_train_edges.append(category_edges)
    if len(city_edges) > 0:
        static_train_edges.append(city_edges)

    static_train = pd.concat(static_train_edges, ignore_index=True)

    train_all = pd.concat([review_train_e, tip_train_e, static_train], ignore_index=True)
    val_all = pd.concat([review_val_e, tip_val_e], ignore_index=True)
    test_all = pd.concat([review_test_e, tip_test_e], ignore_index=True)

    # ------------------------------------------------------------------
    # Save base splits
    # ------------------------------------------------------------------
    def save_split(df: pd.DataFrame, path: Path):
        df[["src", "relation", "dst"]].to_csv(path, sep="\t", index=False, header=False)

    save_split(train_all, out_dir / "train.tsv")
    save_split(val_all, out_dir / "val.tsv")
    save_split(test_all, out_dir / "test.tsv")

    # ------------------------------------------------------------------
    # Save relation-specific splits
    # ------------------------------------------------------------------
    save_split(review_train_e, out_dir / "review_train.tsv")
    save_split(review_val_e, out_dir / "review_val.tsv")
    save_split(review_test_e, out_dir / "review_test.tsv")

    save_split(tip_train_e, out_dir / "tip_train.tsv")
    save_split(tip_val_e, out_dir / "tip_val.tsv")
    save_split(tip_test_e, out_dir / "tip_test.tsv")

    save_split(friend_edges_mapped, out_dir / "friend_train.tsv")

    if len(friend_test_e) > 0:
        save_split(friend_test_e, out_dir / "friend_test.tsv")

    if len(category_edges) > 0:
        save_split(category_edges, out_dir / "in_category_train.tsv")

    if len(city_edges) > 0:
        save_split(city_edges, out_dir / "located_in_train.tsv")

    # ------------------------------------------------------------------
    # Ablation folders (train only)
    # ------------------------------------------------------------------
    ablations = {
        "all": [
            review_train_e,
            tip_train_e,
            friend_edges_mapped,
            category_edges,
            city_edges,
        ],
        "no_tip": [
            review_train_e,
            friend_edges_mapped,
            category_edges,
            city_edges,
        ],
        "no_friend": [
            review_train_e,
            tip_train_e,
            category_edges,
            city_edges,
        ],
        "no_review": [
            tip_train_e,
            friend_edges_mapped,
            category_edges,
            city_edges,
        ],
        "review_only": [review_train_e],
        "tip_only": [tip_train_e],
        "friend_only": [friend_edges_mapped],
        "review_friend": [
            review_train_e,
            friend_edges_mapped,
            category_edges,
            city_edges,
        ],
        "review_tip": [
            review_train_e,
            tip_train_e,
            category_edges,
            city_edges,
        ],
    }

    for name, parts in ablations.items():
        ab_dir = out_dir / f"ablation_{name}"
        ab_dir.mkdir(exist_ok=True)
        train_df = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)
        save_split(train_df, ab_dir / "train.tsv")
        save_split(val_all, ab_dir / "val.tsv")
        save_split(test_all, ab_dir / "test.tsv")

    # ------------------------------------------------------------------
    # Save entities and relations metadata
    # ------------------------------------------------------------------
    entities_rows = []

    entities_rows.extend(
        {
            "entity_id": eid,
            "original_id": uid,
            "entity_type": "user",
        }
        for uid, eid in user2id.items()
    )

    entities_rows.extend(
        {
            "entity_id": eid,
            "original_id": bid,
            "entity_type": "business",
        }
        for bid, eid in biz2id.items()
    )

    entities_rows.extend(
        {
            "entity_id": eid,
            "original_id": c,
            "entity_type": "category",
        }
        for c, eid in cat2id.items()
    )

    entities_rows.extend(
        {
            "entity_id": eid,
            "original_id": c,
            "entity_type": "city",
        }
        for c, eid in city2id.items()
    )

    entities_table = pa.Table.from_pandas(pd.DataFrame(entities_rows))
    pq.write_table(entities_table, out_dir / "entities.parquet")

    relations_table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "relation_id": list(rel2id.values()),
                "name": list(rel2id.keys()),
            }
        )
    )
    pq.write_table(relations_table, out_dir / "relations.parquet")

    report = {
        "dataset": "Yelp Open Dataset",
        "state": state,
        "entities": {
            "users": len(user2id),
            "businesses": len(biz2id),
            "categories": len(cat2id),
            "cities": len(city2id),
            "total": total_entities,
        },
        "relations": rel2id,
        "splits": {
            "train_end": train_end,
            "val_end": val_end,
            "test_after": val_end,
        },
        "edges": {
            "review_train": len(review_train_e),
            "review_val": len(review_val_e),
            "review_test": len(review_test_e),
            "tip_train": len(tip_train_e),
            "tip_val": len(tip_val_e),
            "tip_test": len(tip_test_e),
            "friend_train": len(friend_edges_mapped),
            "friend_test": len(friend_test_e),
            "in_category_train": len(category_edges),
            "located_in_train": len(city_edges),
        },
        "total_train_edges": len(train_all),
    }

    with open(out_dir / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved graph to:", out_dir)


if __name__ == "__main__":
    main()
