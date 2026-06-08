
from __future__ import annotations

import pandas as pd
from typing import Tuple


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str, date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    if date_col not in df.columns:
        raise KeyError(f"temporal_split: нет колонки '{date_col}' в данных")

    if pd.Timestamp(train_end) >= pd.Timestamp(val_end):
        raise ValueError(f"train_end ({train_end}) должен быть раньше val_end ({val_end})")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train = df[df[date_col] <= pd.Timestamp(train_end)]
    val = df[(df[date_col] > pd.Timestamp(train_end)) & (df[date_col] <= pd.Timestamp(val_end))]
    test = df[df[date_col] > pd.Timestamp(val_end)]

    return train, val, test
