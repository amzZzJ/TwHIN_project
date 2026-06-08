
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Relation-type ablations for the Table 2 analog: single-relation models and
# leave-one-out, plus the full graph and pairwise combinations referenced in
# the README's recommended experiments.
DEFAULT_ABLATIONS = [
    "all",
    "review_only",
    "tip_only",
    "friend_only",
    "no_tip",
    "no_friend",
    "no_review",
    "review_friend",
    "review_tip",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", choices=["ns", "sampled_softmax"], default="ns")
    parser.add_argument("--logq", action="store_true")
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    args = parser.parse_args()

    for ab in args.ablations:
        cmd = [sys.executable, "-m", "src.training.train",
               "--config", args.config, "--ablation", ab, "--loss", args.loss]
        if args.logq:
            cmd.append("--logq")
        print("\n" + "=" * 70)
        print("TRAIN ablation:", ab)
        print("=" * 70)
        subprocess.run(cmd, check=True)

    print("\nAll ablations trained. Checkpoints in outputs/checkpoints/")


if __name__ == "__main__":
    main()
