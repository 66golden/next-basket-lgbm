from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np
import pandas as pd


def _safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def _days_between(later_ts: pd.Timestamp, earlier_ts: pd.Timestamp) -> float:
    delta = later_ts - earlier_ts
    return float(delta.total_seconds() / 86400.0)


@dataclass
class QueryBuildResult:
    query_rows_df: pd.DataFrame
    group_sizes: list[int]


class ItemHistoryStats(TypedDict):
    item_ts_dict: dict[int, list[pd.Timestamp]]
    item_last2_count: dict[int, int]
    item_last3_count: dict[int, int]
    item_last5_count: dict[int, int]
    item_in_last1: dict[int, int]
    item_in_last2: dict[int, int]


class SimpleFeatureBuilder:
    def __init__(
        self,
        global_top_items: Optional[np.ndarray] = None,
        max_negatives_per_query: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = True,
        log_every_n_queries: int = 500,
    ):
        self.global_top_items = global_top_items
        self.max_negatives_per_query = max_negatives_per_query
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.log_every_n_queries = int(log_every_n_queries)

        self.item_popularity_: dict[int, int] | None = None
        self.item_user_coverage_: dict[int, int] | None = None
        self.user_histories_: dict[int, pd.DataFrame] | None = None
        self.global_top_items_set_: set[int] | None = None
        self.cooc_counts_: dict[int, dict[int, int]] | None = None
        self.rng_ = np.random.default_rng(self.random_state)

    def fit(self, train_df: pd.DataFrame) -> "SimpleFeatureBuilder":
        train_df = train_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        exploded = (
            train_df.loc[:, ["user_id", "basket"]]
            .explode("basket", ignore_index=True)
            .rename(columns={"basket": "item_id"})
        )
        exploded["item_id"] = exploded["item_id"].astype(int)

        item_pop = (
            exploded.groupby("item_id")
            .size()
            .reset_index(name="cnt")
        )
        self.item_popularity_ = dict(zip(item_pop["item_id"], item_pop["cnt"]))

        item_cov = (
            exploded.groupby("item_id")["user_id"]
            .nunique()
            .reset_index(name="user_cnt")
        )
        self.item_user_coverage_ = dict(zip(item_cov["item_id"], item_cov["user_cnt"]))

        self.user_histories_ = {
            int(user_id): user_df.sort_values("timestamp").reset_index(drop=True).copy()
            for user_id, user_df in train_df.groupby("user_id", sort=False)
        }

        if self.global_top_items is not None:
            self.global_top_items_set_ = set(map(int, self.global_top_items))
        else:
            self.global_top_items_set_ = set()

        # global item-item co-occurrence counts from train baskets
        cooc_counts: dict[int, dict[int, int]] = {}
        baskets = train_df["basket"].tolist()
        total_baskets = len(baskets)

        for idx, basket in enumerate(baskets, start=1):
            if self.verbose and (idx == 1 or idx % 50000 == 0 or idx == total_baskets):
                print(f"[feature_builder.fit] cooc processed {idx}/{total_baskets} baskets")

            basket_unique = sorted(set(map(int, basket)))
            n = len(basket_unique)

            for i in range(n):
                item_i = basket_unique[i]
                if item_i not in cooc_counts:
                    cooc_counts[item_i] = {}

                for j in range(n):
                    if i == j:
                        continue
                    item_j = basket_unique[j]
                    cooc_counts[item_i][item_j] = cooc_counts[item_i].get(item_j, 0) + 1

        self.cooc_counts_ = cooc_counts
        return self

    def _check_is_fitted(self) -> None:
        if (
            self.item_popularity_ is None
            or self.item_user_coverage_ is None
            or self.user_histories_ is None
            or self.cooc_counts_ is None
        ):
            raise ValueError("Feature builder is not fitted. Call fit(train_df) first.")

    def _get_user_history_before_ts(
        self,
        user_id: int,
        target_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        user_df = self.user_histories_.get(int(user_id))

        if user_df is None or len(user_df) == 0:
            return pd.DataFrame(columns=["user_id", "basket", "timestamp"])

        history_df = user_df.loc[user_df["timestamp"] < target_timestamp].copy()
        history_df = history_df.sort_values("timestamp").reset_index(drop=True)
        return history_df

    def _build_item_stats_for_history(
        self,
        user_history_df: pd.DataFrame,
    ) -> ItemHistoryStats:
        item_ts_dict: dict[int, list[pd.Timestamp]] = {}
        item_last2_count: dict[int, int] = {}
        item_last3_count: dict[int, int] = {}
        item_last5_count: dict[int, int] = {}
        item_in_last1: dict[int, int] = {}
        item_in_last2: dict[int, int] = {}

        if len(user_history_df) == 0:
            return {
                "item_ts_dict": item_ts_dict,
                "item_last2_count": item_last2_count,
                "item_last3_count": item_last3_count,
                "item_last5_count": item_last5_count,
                "item_in_last1": item_in_last1,
                "item_in_last2": item_in_last2,
            }

        baskets = user_history_df["basket"].tolist()
        timestamps = user_history_df["timestamp"].tolist()

        n = len(baskets)
        last2_start = max(0, n - 2)
        last3_start = max(0, n - 3)
        last5_start = max(0, n - 5)
        last1_idx = n - 1

        for idx, (basket, ts) in enumerate(zip(baskets, timestamps)):
            basket_unique = set(map(int, basket))

            for item_id in basket_unique:
                item_ts_dict.setdefault(item_id, []).append(ts)

                if idx >= last2_start:
                    item_last2_count[item_id] = item_last2_count.get(item_id, 0) + 1
                    item_in_last2[item_id] = 1
                if idx >= last3_start:
                    item_last3_count[item_id] = item_last3_count.get(item_id, 0) + 1
                if idx >= last5_start:
                    item_last5_count[item_id] = item_last5_count.get(item_id, 0) + 1
                if idx == last1_idx:
                    item_in_last1[item_id] = 1

        return {
            "item_ts_dict": item_ts_dict,
            "item_last2_count": item_last2_count,
            "item_last3_count": item_last3_count,
            "item_last5_count": item_last5_count,
            "item_in_last1": item_in_last1,
            "item_in_last2": item_in_last2,
        }

    def _compute_ui_features_from_stats(
        self,
        item_stats: ItemHistoryStats,
        item_id: int,
        target_timestamp: pd.Timestamp,
    ) -> dict:
        item_ts_dict = item_stats["item_ts_dict"]
        item_last2_count = item_stats["item_last2_count"]
        item_last3_count = item_stats["item_last3_count"]
        item_last5_count = item_stats["item_last5_count"]
        item_in_last1 = item_stats["item_in_last1"]
        item_in_last2 = item_stats["item_in_last2"]

        purchase_ts = item_ts_dict.get(int(item_id), [])
        ui_count_total = len(purchase_ts)

        ui_count_last_2_baskets = int(item_last2_count.get(int(item_id), 0))
        ui_count_last_3_baskets = int(item_last3_count.get(int(item_id), 0))
        ui_count_last_5_baskets = int(item_last5_count.get(int(item_id), 0))
        ui_in_last_basket = int(item_in_last1.get(int(item_id), 0))
        ui_in_last_2_baskets = int(item_in_last2.get(int(item_id), 0))

        if ui_count_total == 0:
            ui_last_purchase_days_ago = -1.0
            ui_second_last_purchase_days_ago = -1.0
            ui_mean_gap_days = 0.0
            ui_std_gap_days = 0.0
            ui_gap_from_typical = 0.0
        else:
            ui_last_purchase_days_ago = _days_between(target_timestamp, purchase_ts[-1])

            if ui_count_total >= 2:
                ui_second_last_purchase_days_ago = _days_between(target_timestamp, purchase_ts[-2])

                gaps = [
                    _days_between(purchase_ts[i], purchase_ts[i - 1])
                    for i in range(1, len(purchase_ts))
                ]
                ui_mean_gap_days = _safe_mean(gaps)
                ui_std_gap_days = _safe_std(gaps)
                ui_gap_from_typical = (
                    ui_last_purchase_days_ago / ui_mean_gap_days if ui_mean_gap_days > 0 else 0.0
                )
            else:
                ui_second_last_purchase_days_ago = -1.0
                ui_mean_gap_days = 0.0
                ui_std_gap_days = 0.0
                ui_gap_from_typical = 0.0

        return {
            "ui_count_total": int(ui_count_total),
            "ui_count_last_2_baskets": int(ui_count_last_2_baskets),
            "ui_count_last_3_baskets": int(ui_count_last_3_baskets),
            "ui_count_last_5_baskets": int(ui_count_last_5_baskets),
            "ui_in_last_basket": int(ui_in_last_basket),
            "ui_in_last_2_baskets": int(ui_in_last_2_baskets),
            "ui_last_purchase_days_ago": float(ui_last_purchase_days_ago),
            "ui_second_last_purchase_days_ago": float(ui_second_last_purchase_days_ago),
            "ui_mean_gap_days": float(ui_mean_gap_days),
            "ui_std_gap_days": float(ui_std_gap_days),
            "ui_gap_from_typical": float(ui_gap_from_typical),
        }

    def _compute_user_features(
        self,
        user_history_df: pd.DataFrame,
        target_timestamp: pd.Timestamp,
    ) -> dict:
        if len(user_history_df) == 0:
            return {
                "user_baskets_count_before_t": 0,
                "user_items_unique_before_t": 0,
                "user_avg_basket_size": 0.0,
                "user_std_basket_size": 0.0,
                "user_days_since_last_basket": -1.0,
                "user_avg_gap_between_baskets": 0.0,
                "user_std_gap_between_baskets": 0.0,
                "last_basket_size": 0.0,
            }

        basket_sizes = [len(b) for b in user_history_df["basket"]]
        all_items = np.concatenate([np.asarray(b, dtype=int) for b in user_history_df["basket"]])

        if len(user_history_df) >= 2:
            ts_list = user_history_df["timestamp"].tolist()
            basket_gaps = [
                _days_between(ts_list[i], ts_list[i - 1])
                for i in range(1, len(ts_list))
            ]
        else:
            basket_gaps = []

        last_basket_ts = user_history_df.iloc[-1]["timestamp"]
        last_basket_size = len(user_history_df.iloc[-1]["basket"])

        return {
            "user_baskets_count_before_t": int(len(user_history_df)),
            "user_items_unique_before_t": int(len(np.unique(all_items))),
            "user_avg_basket_size": float(np.mean(basket_sizes)),
            "user_std_basket_size": float(np.std(basket_sizes)) if len(basket_sizes) > 1 else 0.0,
            "user_days_since_last_basket": float(_days_between(target_timestamp, last_basket_ts)),
            "user_avg_gap_between_baskets": float(_safe_mean(basket_gaps)),
            "user_std_gap_between_baskets": float(_safe_std(basket_gaps)),
            "last_basket_size": float(last_basket_size),
        }

    def _compute_item_features(
        self,
        item_id: int,
    ) -> dict:
        self._check_is_fitted()

        item_id = int(item_id)
        item_global_popularity = int(self.item_popularity_.get(item_id, 0))
        item_user_coverage = int(self.item_user_coverage_.get(item_id, 0))
        item_is_global_top = int(item_id in self.global_top_items_set_)

        return {
            "item_global_popularity": item_global_popularity,
            "item_user_coverage": item_user_coverage,
            "item_is_global_top": item_is_global_top,
        }

    def _compute_last_basket_context_features(
        self,
        user_history_df: pd.DataFrame,
        item_id: int,
    ) -> dict:
        self._check_is_fitted()

        if len(user_history_df) == 0:
            return {
                "cooc_last_basket_sum_log": 0.0,
                "cooc_last_basket_max_log": 0.0,
                "cooc_last_basket_mean_log": 0.0,
                "candidate_in_last_basket": 0,
            }

        last_basket = list(map(int, user_history_df.iloc[-1]["basket"]))
        item_id = int(item_id)

        cooc_values = []
        for last_item in last_basket:
            if last_item == item_id:
                continue
            cooc_val = self.cooc_counts_.get(last_item, {}).get(item_id, 0)
            cooc_values.append(float(cooc_val))

        if len(cooc_values) == 0:
            cooc_sum = 0.0
            cooc_max = 0.0
            cooc_mean = 0.0
        else:
            cooc_sum = float(np.sum(cooc_values))
            cooc_max = float(np.max(cooc_values))
            cooc_mean = float(np.mean(cooc_values))

        return {
            "cooc_last_basket_sum_log": float(np.log1p(cooc_sum)),
            "cooc_last_basket_max_log": float(np.log1p(cooc_max)),
            "cooc_last_basket_mean_log": float(np.log1p(cooc_mean)),
            "candidate_in_last_basket": int(item_id in set(last_basket)),
        }

    def _sample_negatives(
        self,
        candidates: np.ndarray,
        positives: set[int],
    ) -> np.ndarray:
        if self.max_negatives_per_query is None:
            return candidates

        negatives = np.asarray([x for x in candidates if int(x) not in positives], dtype=int)
        positives_arr = np.asarray(sorted(list(positives)), dtype=int)

        if len(negatives) <= self.max_negatives_per_query:
            sampled_negatives = negatives
        else:
            sampled_negatives = self.rng_.choice(
                negatives,
                size=self.max_negatives_per_query,
                replace=False,
            )

        result = np.unique(np.concatenate([positives_arr, sampled_negatives]))
        return result

    def build_dataset(
        self,
        states_df: pd.DataFrame,
        candidate_generator,
        train_mode: bool,
    ) -> QueryBuildResult:
        self._check_is_fitted()

        rows = []
        group_sizes = []

        total_queries = len(states_df)

        for q_idx, (_, state) in enumerate(states_df.iterrows(), start=1):
            if self.verbose and (q_idx == 1 or q_idx % self.log_every_n_queries == 0 or q_idx == total_queries):
                print(f"[build_dataset] processed {q_idx}/{total_queries} queries")

            query_id = int(state["query_id"])
            user_id = int(state["user_id"])
            target_timestamp = pd.Timestamp(state["target_timestamp"])
            target_basket = list(state["target_basket"])

            user_history_df = self._get_user_history_before_ts(
                user_id=user_id,
                target_timestamp=target_timestamp,
            )

            candidates = candidate_generator.generate_for_history(user_history_df)
            positives = set(map(int, target_basket))

            if train_mode:
                candidates = self._sample_negatives(candidates=candidates, positives=positives)

            user_features = self._compute_user_features(
                user_history_df=user_history_df,
                target_timestamp=target_timestamp,
            )

            item_stats = self._build_item_stats_for_history(user_history_df)

            query_rows = []
            for item_id in candidates:
                item_id = int(item_id)

                ui_features = self._compute_ui_features_from_stats(
                    item_stats=item_stats,
                    item_id=item_id,
                    target_timestamp=target_timestamp,
                )
                item_features = self._compute_item_features(item_id=item_id)
                context_features = self._compute_last_basket_context_features(
                    user_history_df=user_history_df,
                    item_id=item_id,
                )

                row = {
                    "query_id": query_id,
                    "user_id": user_id,
                    "item_id": item_id,
                    "target": int(item_id in positives),
                    "target_timestamp": target_timestamp,
                    "source": state["source"],
                }
                row.update(user_features)
                row.update(ui_features)
                row.update(item_features)
                row.update(context_features)

                query_rows.append(row)

            query_rows_df = pd.DataFrame(query_rows)
            query_rows_df = query_rows_df.sort_values(["query_id", "item_id"]).reset_index(drop=True)

            rows.append(query_rows_df)
            group_sizes.append(len(query_rows_df))

        full_df = pd.concat(rows, axis=0, ignore_index=True)
        return QueryBuildResult(query_rows_df=full_df, group_sizes=group_sizes)