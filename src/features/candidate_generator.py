from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class SimpleCandidateGenerator:
    """
    Первый безопасный генератор кандидатов:
    1) все товары из истории пользователя;
    2) top-M глобально популярных товаров из train.

    Это correctness-first baseline.
    """

    def __init__(self, global_top_m: int = 100):
        self.global_top_m = int(global_top_m)
        self.global_top_items_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "SimpleCandidateGenerator":
        exploded = (
            train_df[["user_id", "basket"]]
            .explode("basket", ignore_index=True)
            .rename(columns={"basket": "item_id"})
        )

        exploded["item_id"] = exploded["item_id"].astype(int)

        pop = (
            exploded.groupby("item_id")
            .size()
            .reset_index(name="cnt")
            .sort_values(["cnt", "item_id"], ascending=[False, True])
            .reset_index(drop=True)
        )

        self.global_top_items_ = pop["item_id"].head(self.global_top_m).to_numpy(dtype=int)
        return self

    def _check_is_fitted(self) -> None:
        if self.global_top_items_ is None:
            raise ValueError("Candidate generator is not fitted. Call fit(train_df) first.")

    def generate_for_history(self, user_history_df: pd.DataFrame) -> np.ndarray:
        """
        user_history_df — история одного пользователя до target_timestamp.
        """
        self._check_is_fitted()

        history_items = []
        if len(user_history_df) > 0:
            history_items = (
                user_history_df["basket"]
                .explode()
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

        candidates = np.unique(
            np.concatenate(
                [
                    np.asarray(history_items, dtype=int),
                    self.global_top_items_,
                ]
            )
        )
        return candidates

    def generate_many(
        self,
        histories: Iterable[pd.DataFrame],
    ) -> list[np.ndarray]:
        self._check_is_fitted()
        return [self.generate_for_history(h) for h in histories]