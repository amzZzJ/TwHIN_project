
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from recsys.TwHIN_project.twhin_yelp_3.src.data.dataset import TripleDataset, load_num_entities, load_num_relations
from recsys.TwHIN_project.twhin_yelp_3.src.data.sampler import NegativeSampler
from recsys.TwHIN_project.twhin_yelp_3.src.losses.ns_loss import negative_sampling_loss
from recsys.TwHIN_project.twhin_yelp_3.src.losses.sampled_softmax import sampled_softmax_loss
from recsys.TwHIN_project.twhin_yelp_3.src.models.transe import TransE


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ns_loss_branch(model, h, r, t, sampler, neg_k, corrupt, generator=None):
    """Negative-sampling loss with head/tail corruption (paper Eq. 2).

    N(s, r, t) = {(s, r, t')} U {(s', r, t)} -> corrupt the tail, the head, or both.
    When corrupt == 'both', neg_k is split evenly between tail and head corruptions
    so the total number of negatives stays neg_k.
    """
    pos = model.score(h, r, t)
    B = h.size(0)

    neg_scores = []
    if corrupt in ("tail", "both"):
        k_tail = neg_k // 2 if corrupt == "both" else neg_k
        if k_tail > 0:
            neg_t = sampler.sample((B, k_tail), h.device, generator=generator)
            neg_scores.append(model.tail_scores(h, r, neg_t))
    if corrupt in ("head", "both"):
        k_head = neg_k - (neg_k // 2) if corrupt == "both" else neg_k
        if k_head > 0:
            neg_h = sampler.sample((B, k_head), h.device, generator=generator)
            neg_scores.append(model.head_scores(neg_h, r, t))

    neg = torch.cat(neg_scores, dim=1)
    return negative_sampling_loss(pos, neg)


def softmax_loss_branch(model, h, r, t, sampler, neg_k, use_logq, generator=None):
    pos = model.score(h, r, t)
    cand_t = sampler.sample((h.size(0), neg_k), h.device, generator=generator)
    neg = model.tail_scores(h, r, cand_t)

    if use_logq:
        log_q = sampler.log_q(cand_t, h.device)
        pos_log_q = sampler.log_q(t[:, None], h.device)
        log_q = torch.cat([pos_log_q, log_q], dim=1)
    else:
        log_q = None

    return sampled_softmax_loss(pos, neg, log_q)


@torch.no_grad()
def evaluate(model: TransE, loader: DataLoader, device: torch.device, loss_name: str, sampler: NegativeSampler, neg_k: int, use_logq: bool, corrupt: str = "both") -> float:
    model.eval()

    losses = []

    gen = torch.Generator(device=device)
    gen.manual_seed(12345)

    for h, r, t in loader:
        h = h.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        if loss_name == "ns":
            loss = ns_loss_branch(model, h, r, t, sampler, neg_k, corrupt, generator=gen)
        else:
            loss = softmax_loss_branch(model, h, r, t, sampler, neg_k, use_logq, generator=gen)

        losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", choices=["ns", "sampled_softmax"], default="ns")
    parser.add_argument("--logq", action="store_true")
    parser.add_argument("--ablation", default="all", help="ablation folder suffix, e.g. all, no_tip, review_only")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    out_dir = Path(cfg["out_dir"])
    ablation_dir = out_dir / f"ablation_{args.ablation}"

    train_path = ablation_dir / "train.tsv"
    val_path = ablation_dir / "val.tsv"

    num_entities = load_num_entities(out_dir)
    num_relations = load_num_relations(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TripleDataset(str(train_path))
    val_ds = TripleDataset(str(val_path))

    batch_size = int(cfg["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )

    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=int(cfg["embedding_dim"]),
        score_fn=str(cfg.get("score_fn", "dot")),
    ).to(device)

    opt_name = str(cfg.get("optimizer", "adagrad")).lower()
    if opt_name == "adagrad":
        # Paper optimizes with Adagrad (Sec 4.1).
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    else:
        raise ValueError(f"unknown optimizer {opt_name}")

    amp_enabled = bool(cfg["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    neg_k = int(cfg["negative_samples"])
    corrupt = str(cfg["negative_sampling"].get("corrupt", "both"))

    sampler = NegativeSampler(
        train_tsv=str(train_path),
        num_entities=num_entities,
        strategy=str(cfg["negative_sampling"]["strategy"]),
        power=float(cfg["negative_sampling"]["power"]),
    )

    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.ablation}__{args.loss}" + ("__logq" if args.logq else "")
    ckpt_path = ckpt_dir / f"{tag}.pt"

    best_val = float("inf")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()

        running = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")

        for h, r, t in pbar:
            h = h.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            model.normalize()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                if args.loss == "ns":
                    loss = ns_loss_branch(model, h, r, t, sampler, neg_k, corrupt)
                else:
                    loss = softmax_loss_branch(model, h, r, t, sampler, neg_k, args.logq)

            scaler.scale(loss).backward()

            if float(cfg["grad_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))

            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_name=args.loss,
            sampler=sampler,
            neg_k=neg_k,
            use_logq=args.logq,
            corrupt=corrupt,
        )

        print(f"epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss

            payload = {
                "state_dict": model.state_dict(),
                "num_entities": num_entities,
                "num_relations": num_relations,
                "embedding_dim": int(cfg["embedding_dim"]),
                "score_fn": str(cfg.get("score_fn", "dot")),
                "loss": args.loss,
                "logq": bool(args.logq),
                "ablation": args.ablation,
                "val_loss": val_loss,
            }

            torch.save(payload, ckpt_path)
            torch.save(payload, ckpt_dir / "best.pt")

            print(f"saved best checkpoint -> {ckpt_path.name}")

    print("done")


if __name__ == "__main__":
    main()
