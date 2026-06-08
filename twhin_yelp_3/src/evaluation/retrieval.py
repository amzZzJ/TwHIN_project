
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import MiniBatchKMeans

from recsys.TwHIN_project.twhin_yelp_3.src.data.dataset import load_entity_table
from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.common import entity_embeddings, load_model


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_faiss_index(vectors: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(vectors.shape[1])
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    index.add(vectors.astype(np.float32))
    return index


def metrics_from_ranked(ranked_lists: Dict[int, List[int]], gt: Dict[int, set], ks: List[int]):
    hit = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    mrrs = []
    for u, positives in gt.items():
        if u not in ranked_lists:
            continue
        cand = ranked_lists[u]
        for k in ks:
            topk = set(cand[:k])
            hit[k].append(1.0 if topk & positives else 0.0)
            rec[k].append(len(topk & positives) / max(1, len(positives)))
        rr = 0.0
        for rank, eid in enumerate(cand, start=1):
            if eid in positives:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    out = {"queries": len(mrrs), "MRR": float(np.mean(mrrs)) if mrrs else 0.0}
    for k in ks:
        out[f"HitRate@{k}"] = float(np.mean(hit[k])) if hit[k] else 0.0
        out[f"Recall@{k}"] = float(np.mean(rec[k])) if rec[k] else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", choices=["unimodal", "mixture", "both"], default="both")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["out_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)
    entities = load_entity_table(out_dir)

    user_ids = entities.loc[entities.entity_type == "user", "entity_id"].astype(int).to_numpy()
    biz_ids = entities.loc[entities.entity_type == "business", "entity_id"].astype(int).to_numpy()

    # L2-normalized embeddings so inner product == cosine (ANN setting).
    emb = entity_embeddings(model, l2_normalize=True)
    user_vecs = emb[user_ids]
    user_pos = {int(eid): i for i, eid in enumerate(user_ids)}

    index = build_faiss_index(user_vecs, use_gpu=bool(cfg["eval"].get("faiss_gpu", False)))

    # ground truth + known (training) friends to filter
    friend_test_path = out_dir / "friend_test.tsv"
    if not friend_test_path.exists():
        print("friend_test.tsv not found; retrieval evaluation skipped")
        return
    gt_df = pd.read_csv(friend_test_path, sep="\t", header=None, names=["src", "rel", "dst"])
    gt: Dict[int, set] = {}
    for row in gt_df.itertuples(index=False):
        gt.setdefault(int(row.src), set()).add(int(row.dst))

    known: Dict[int, set] = {}
    friend_train_path = out_dir / "friend_train.tsv"
    if friend_train_path.exists():
        tr = pd.read_csv(friend_train_path, sep="\t", header=None, names=["src", "rel", "dst"])
        for row in tr.itertuples(index=False):
            known.setdefault(int(row.src), set()).add(int(row.dst))
            known.setdefault(int(row.dst), set()).add(int(row.src))

    ks: List[int] = list(cfg["eval"]["recall_ks"])
    max_k = max(ks)

    # ---------------- unimodal ----------------
    def unimodal_lists():
        ranked = {}
        for u in gt:
            if u not in user_pos:
                continue
            q = user_vecs[user_pos[u]][None, :]
            _, I = index.search(q, max_k + len(known.get(u, ())) + 1)
            seen = known.get(u, set())
            ranked[u] = [int(x) for x in user_ids[I[0]] if x != u and x not in seen][:max_k]
        return ranked

    # ---------------- mixture-of-embeddings (paper Sec 4.4 / Eq. 3) ----------------
    def mixture_lists():
        mcfg = cfg.get("mixture", {})
        n_clusters = int(mcfg.get("n_clusters", 50))
        top_m = int(mcfg.get("top_m", 3))
        kmeans_iters = int(mcfg.get("kmeans_iters", 25))

        # cluster the universe of non-user (business) targets
        biz_vecs = emb[biz_ids]
        n_clusters = min(n_clusters, len(biz_ids))
        km = MiniBatchKMeans(n_clusters=n_clusters, max_iter=kmeans_iters,
                             random_state=int(cfg.get("seed", 42)), n_init=3)
        labels = km.fit_predict(biz_vecs)
        centroids = km.cluster_centers_.astype(np.float32)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
        biz_cluster = {int(b): int(c) for b, c in zip(biz_ids, labels)}

        # user -> cluster engagement counts from training engagements (review + tip)
        eng_frames = []
        for name in ("review_train.tsv", "tip_train.tsv"):
            p = out_dir / name
            if p.exists():
                eng_frames.append(pd.read_csv(p, sep="\t", header=None, names=["src", "rel", "dst"]))
        if not eng_frames:
            return {}
        eng = pd.concat(eng_frames, ignore_index=True)
        eng["cluster"] = eng["dst"].map(biz_cluster)
        eng = eng.dropna(subset=["cluster"])
        eng["cluster"] = eng["cluster"].astype(int)

        user_cluster_counts: Dict[int, Dict[int, int]] = {}
        for u, c in zip(eng["src"].astype(int), eng["cluster"]):
            d = user_cluster_counts.setdefault(u, {})
            d[c] = d.get(c, 0) + 1

        ranked = {}
        for u in gt:
            if u not in user_pos:
                continue
            counts = user_cluster_counts.get(u)
            if not counts:
                # fall back to the user's own embedding when no engagement history
                q = user_vecs[user_pos[u]][None, :]
                _, I = index.search(q, max_k + 1)
                seen = known.get(u, set())
                ranked[u] = [int(x) for x in user_ids[I[0]] if x != u and x not in seen][:max_k]
                continue
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_m]
            total = sum(w for _, w in top)
            seen = known.get(u, set()) | {u}
            merged: List[int] = []
            for cl, w in top:
                budget = max(1, int(round(max_k * w / total)))
                q = centroids[cl][None, :]
                _, I = index.search(q, budget + len(seen) + 1)
                for x in user_ids[I[0]]:
                    x = int(x)
                    if x in seen or x in merged:
                        continue
                    merged.append(x)
                    if len([m for m in merged]) >= max_k:
                        break
            ranked[u] = merged[:max_k]
        return ranked

    results = {}
    if args.mode in ("unimodal", "both"):
        results["unimodal"] = metrics_from_ranked(unimodal_lists(), gt, ks)
    if args.mode in ("mixture", "both"):
        results["mixture"] = metrics_from_ranked(mixture_lists(), gt, ks)

    # sanity baselines (random / most-popular)
    rng = np.random.default_rng(0)
    deg = pd.concat([
        pd.read_csv(friend_train_path, sep="\t", header=None, names=["src", "rel", "dst"])["src"],
        pd.read_csv(friend_train_path, sep="\t", header=None, names=["src", "rel", "dst"])["dst"],
    ]).value_counts()
    most_popular = [int(x) for x in deg.index[: max_k * 3]]
    rand_lists, pop_lists = {}, {}
    for u in gt:
        if u not in user_pos:
            continue
        seen = known.get(u, set()) | {u}
        rand_lists[u] = [int(x) for x in rng.choice(user_ids, size=max_k, replace=False)]
        pop_lists[u] = [x for x in most_popular if x not in seen][:max_k]
    results["random"] = metrics_from_ranked(rand_lists, gt, ks)
    results["most_popular"] = metrics_from_ranked(pop_lists, gt, ks)

    # print
    print("Candidate generation (friend recommendation)  --  Table 1 analog")
    header = f"  {'method':>13} | " + " | ".join(f"R@{k:<3}" for k in ks) + " |  MRR"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in ["unimodal", "mixture", "most_popular", "random"]:
        if m not in results:
            continue
        r = results[m]
        cells = " | ".join(f"{r[f'Recall@{k}']*100:5.2f}" for k in ks)
        print(f"  {m:>13} | {cells} | {r['MRR']:.4f}")

    if "unimodal" in results and "mixture" in results:
        u10 = results["unimodal"][f"Recall@{ks[0]}"]
        m10 = results["mixture"][f"Recall@{ks[0]}"]
        if u10 > 0:
            print(f"\n  mixture vs unimodal Recall@{ks[0]} lift: {100*(m10-u10)/u10:+.1f}%")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {args.json_out}")


if __name__ == "__main__":
    main()
