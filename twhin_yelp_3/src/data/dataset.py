
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class TripleDataset(Dataset):
    def __init__(self, tsv_path: str):
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["src", "rel", "dst"])
        self.h = torch.tensor(df["src"].values, dtype=torch.long)
        self.r = torch.tensor(df["rel"].values, dtype=torch.long)
        self.t = torch.tensor(df["dst"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.h[idx], self.r[idx], self.t[idx]


def load_num_entities(out_dir: str) -> int:
    table = pq.read_table(Path(out_dir) / "entities.parquet")
    return int(table.num_rows)


def load_num_relations(out_dir: str) -> int:
    table = pq.read_table(Path(out_dir) / "relations.parquet")
    return int(table.num_rows)


def load_relation_map(out_dir: str) -> dict:
    table = pq.read_table(Path(out_dir) / "relations.parquet").to_pandas()
    return {row["name"]: int(row["relation_id"]) for _, row in table.iterrows()}


def load_entity_table(out_dir: str) -> pd.DataFrame:
    return pq.read_table(Path(out_dir) / "entities.parquet").to_pandas()
