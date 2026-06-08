
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from recsys.TwHIN_project.twhin_yelp_3.src.models.transe import TransE


def load_model(ckpt_path: str, device: torch.device) -> TransE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        embedding_dim=ckpt["embedding_dim"],
        score_fn=ckpt.get("score_fn", "dot"),
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def entity_embeddings(model: TransE, l2_normalize: bool = False) -> np.ndarray:
    emb = model.entity_emb.weight.detach().cpu().numpy().astype(np.float32)
    if l2_normalize:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    return emb


def relation_vector(model: TransE, rel_id: int) -> np.ndarray:
    return model.relation_emb.weight.detach().cpu().numpy().astype(np.float32)[rel_id]
