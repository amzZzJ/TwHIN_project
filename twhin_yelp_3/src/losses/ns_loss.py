
from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_sampling_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    pos_loss = F.logsigmoid(pos_score)
    neg_loss = F.logsigmoid(-neg_score).mean(dim=1)
    return -(pos_loss + neg_loss).mean()
