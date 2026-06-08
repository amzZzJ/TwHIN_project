
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """Translating-embedding encoder used by TwHIN.

    The TwHIN paper (El-Kishky et al., KDD 2022) scores a triple (s, r, t) with
    a *dot product* between the translated source and the target (Eq. 1):

        f(s, r, t) = (theta_s + theta_r)^T theta_t

    This is the default here (score_fn="dot"). The classic distance-based TransE
    score  f = -||theta_s + theta_r - theta_t||_2  is also available
    (score_fn="l2") for comparison, but it does NOT match the paper.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        score_fn: str = "dot",
    ):
        super().__init__()

        if score_fn not in ("dot", "l2"):
            raise ValueError(f"score_fn must be 'dot' or 'l2', got {score_fn}")
        self.score_fn = score_fn

        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def normalize(self):
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)

    def _score(self, eh, er, et):
        # eh, er, et broadcastable to [..., d]
        if self.score_fn == "dot":
            # (theta_s + theta_r)^T theta_t  -- paper Eq. 1
            return ((eh + er) * et).sum(dim=-1)
        # negative L2 translation distance (classic TransE, higher is better)
        return -torch.norm(eh + er - et, p=2, dim=-1)

    def score(self, h, r, t):
        eh = self.entity_emb(h)
        er = self.relation_emb(r)
        et = self.entity_emb(t)
        return self._score(eh, er, et)

    def tail_scores(self, h, r, candidate_t):
        # h: [B], r: [B], candidate_t: [B, K]
        eh = self.entity_emb(h)[:, None, :]
        er = self.relation_emb(r)[:, None, :]
        et = self.entity_emb(candidate_t)
        return self._score(eh, er, et)

    def head_scores(self, candidate_h, r, t):
        # candidate_h: [B, K], r: [B], t: [B]
        eh = self.entity_emb(candidate_h)
        er = self.relation_emb(r)[:, None, :]
        et = self.entity_emb(t)[:, None, :]
        return self._score(eh, er, et)
