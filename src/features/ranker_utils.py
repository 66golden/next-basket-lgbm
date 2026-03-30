from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "user_baskets_count_before_t",
    "user_items_unique_before_t",
    "user_avg_basket_size",
    "user_std_basket_size",
    "user_days_since_last_basket",
    "user_avg_gap_between_baskets",
    "user_std_gap_between_baskets",
    "last_basket_size",
    "ui_count_total",
    "ui_count_last_2_baskets",
    "ui_count_last_3_baskets",
    "ui_count_last_5_baskets",
    "ui_in_last_basket",
    "ui_in_last_2_baskets",
    "ui_last_purchase_days_ago",
    "ui_second_last_purchase_days_ago",
    "ui_mean_gap_days",
    "ui_std_gap_days",
    "ui_gap_from_typical",
    "item_global_popularity",
    "item_user_coverage",
    "item_is_global_top",
    "cooc_last_basket_sum_log",
    "cooc_last_basket_max_log",
    "cooc_last_basket_mean_log",
    "candidate_in_last_basket",
]


META_COLUMNS = [
    "query_id",
    "user_id",
    "item_id",
    "target",
    "target_timestamp",
    "source",
]


def prepare_lgbm_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = set(FEATURE_COLUMNS + ["target"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.loc[:, FEATURE_COLUMNS].copy()
    y = df["target"].to_numpy(dtype=int)

    for col in x.columns:
        x[col] = x[col].astype(float)

    return x, y