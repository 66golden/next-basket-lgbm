from __future__ import annotations

from typing import Optional

import pandas as pd


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"user_id", "basket", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    out = df.copy()
    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return out


def build_train_prefix_states(
    train_df: pd.DataFrame,
    min_history_baskets: int = 1,
    max_states_per_user: Optional[int] = None,
) -> pd.DataFrame:
    """
    Строит train-состояния из train_df.

    Для пользователя с корзинами:
        B1, B2, B3, B4
    строим состояния:
        history=[B1]         -> target=B2
        history=[B1, B2]     -> target=B3
        history=[B1, B2, B3] -> target=B4

    Возвращает таблицу состояний:
        query_id
        user_id
        target_timestamp
        target_basket
        history_baskets_count
        source
    """
    train_df = _ensure_sorted(train_df)

    rows = []
    query_id = 0

    for user_id, user_df in train_df.groupby("user_id", sort=False):
        user_df = user_df.sort_values("timestamp").reset_index(drop=True)
        n = len(user_df)

        # target basket index starts from min_history_baskets
        start_idx = min_history_baskets
        end_idx = n

        candidate_target_indices = list(range(start_idx, end_idx))

        if max_states_per_user is not None and len(candidate_target_indices) > max_states_per_user:
            # берем последние max_states_per_user состояний
            candidate_target_indices = candidate_target_indices[-max_states_per_user:]

        for target_idx in candidate_target_indices:
            target_row = user_df.iloc[target_idx]

            rows.append(
                {
                    "query_id": query_id,
                    "user_id": int(user_id),
                    "target_timestamp": target_row["timestamp"],
                    "target_basket": list(target_row["basket"]),
                    "history_baskets_count": int(target_idx),
                    "source": "train_prefix",
                }
            )
            query_id += 1

    states_df = pd.DataFrame(rows)

    if len(states_df) == 0:
        raise ValueError("No train prefix states were created. Check min_history_baskets.")

    return states_df


def build_eval_states(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Строит состояния для validation/test.

    История берется из train_df.
    Целевая корзина и timestamp берутся из target_df.

    Ожидается, что target_df содержит по одной целевой корзине на пользователя.
    """
    if source_name not in {"validation", "test"}:
        raise ValueError("source_name must be 'validation' or 'test'")

    train_df = _ensure_sorted(train_df)
    target_df = _ensure_sorted(target_df)

    history_counts = (
        train_df.groupby("user_id", as_index=False)
        .size()
        .rename(columns={"size": "history_baskets_count"})
    )

    states_df = target_df.rename(
        columns={
            "timestamp": "target_timestamp",
            "basket": "target_basket",
        }
    ).copy()

    states_df = states_df.merge(history_counts, on="user_id", how="left")
    states_df["history_baskets_count"] = states_df["history_baskets_count"].fillna(0).astype(int)
    states_df["source"] = source_name
    states_df = states_df.reset_index(drop=True)
    states_df["query_id"] = range(len(states_df))

    states_df = states_df[
        [
            "query_id",
            "user_id",
            "target_timestamp",
            "target_basket",
            "history_baskets_count",
            "source",
        ]
    ].copy()

    return states_df