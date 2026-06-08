
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


class NegativeSampler:
    def __init__(self, train_tsv: str, num_entities: int, strategy: str = "frequency", power: float = 0.75):
        self.num_entities = int(num_entities)
        self.strategy = strategy

        if strategy == "uniform":
            self.prob = None
            return

        df = pd.read_csv(train_tsv, sep="\t", header=None, names=["src", "rel", "dst"])
        counts = Counter(df["src"].tolist() + df["dst"].tolist())

        freq = np.ones(self.num_entities, dtype=np.float64)
        for eid, c in counts.items():
            if 0 <= int(eid) < self.num_entities:
                freq[int(eid)] = float(c)

        freq = np.power(freq, power)
        self.prob = torch.tensor(freq / freq.sum(), dtype=torch.float32)

    @torch.no_grad()
    def sample(self, shape, device, generator=None) -> torch.Tensor:
        if self.strategy == "uniform":
            return torch.randint(0, self.num_entities, shape, device=device, generator=generator)

        return torch.multinomial(self.prob.to(device), int(np.prod(shape)), replacement=True, generator=generator).view(*shape)

    @torch.no_grad()
    def log_q(self, ids: torch.Tensor, device) -> torch.Tensor:
        if self.strategy == "uniform":
            return torch.full_like(ids, fill_value=-np.log(self.num_entities), dtype=torch.float32, device=device)

        p = self.prob.to(device)[ids]
        return torch.log(p + 1e-12)
