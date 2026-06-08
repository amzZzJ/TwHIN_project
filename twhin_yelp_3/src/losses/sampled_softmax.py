
from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_softmax_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, log_q: torch.Tensor | None = None) -> torch.Tensor:
    # pos_score: [B]
    # neg_score: [B, K]
    logits = torch.cat([pos_score[:, None], neg_score], dim=1)

    if log_q is not None:
        logits = logits - log_q

    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
