
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.common import load_model
from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.engagement import run_engagement
from recsys.TwHIN_project.twhin_yelp_3.src.evaluation.personalized import run_personalized


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", default="ns")
    parser.add_argument("--logq", action="store_true")
    parser.add_argument("--ablations", nargs="*", default=[
        "all", "review_only", "tip_only", "friend_only",
        "no_tip", "no_friend", "no_review", "review_friend", "review_tip",
    ])
    parser.add_argument("--json_out", default="outputs/ablation_report.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("outputs/checkpoints")

    rows = {}
    for ab in args.ablations:
        tag = f"{ab}__{args.loss}" + ("__logq" if args.logq else "")
        ckpt = ckpt_dir / f"{tag}.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint for {ab} ({ckpt.name})")
            continue
        model = load_model(str(ckpt), device)
        rev = run_engagement(cfg, model, relation="review", verbose=False)
        tip = run_engagement(cfg, model, relation="tip", verbose=False)
        per = run_personalized(cfg, model, verbose=False)
        rows[ab] = {
            "review_RCE": rev.get("RCE"),
            "review_ROC": rev.get("ROC-AUC"),
            "tip_RCE": tip.get("RCE"),
            "tip_ROC": tip.get("ROC-AUC"),
            "search_MAP": per.get("MAP"),
            "search_ROC": per.get("avg_ROC"),
        }

    # ---- Table 2 analog ----
    print("\nRelation ablation  --  Table 2 analog (engagement RCE / ROC, search MAP / ROC)")
    cols = ["review_RCE", "review_ROC", "tip_RCE", "tip_ROC", "search_MAP", "search_ROC"]
    head = f"  {'ablation':>14} | " + " | ".join(f"{c:>10}" for c in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for ab, r in rows.items():
        cells = " | ".join(("   n/a   " if r[c] is None else f"{r[c]:10.4f}") for c in cols)
        print(f"  {ab:>14} | {cells}")

    # ---- low/high coverage claim ----
    print("\nLow- vs high-coverage co-training (paper Sec 4.3 / 6.2 claim):")

    def fmt(ab, key):
        return None if ab not in rows or rows[ab].get(key) is None else rows[ab][key]

    tip_solo = fmt("tip_only", "tip_RCE")
    tip_joint = fmt("all", "tip_RCE")
    rev_solo = fmt("review_only", "review_RCE")
    rev_joint = fmt("all", "review_RCE")
    if tip_solo is not None and tip_joint is not None:
        print(f"  tip (low-coverage)  RCE: solo={tip_solo:.3f}  joint(all)={tip_joint:.3f}  "
              f"delta={tip_joint - tip_solo:+.3f}  -> expect POSITIVE (low gains from joint)")
    if rev_solo is not None and rev_joint is not None:
        print(f"  review (high-cov.)  RCE: solo={rev_solo:.3f}  joint(all)={rev_joint:.3f}  "
              f"delta={rev_joint - rev_solo:+.3f}  -> expect ~0/small (high not helped by low)")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {args.json_out}")


if __name__ == "__main__":
    main()
