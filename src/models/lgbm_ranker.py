from __future__ import annotations

from collections import defaultdict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps

from src.dataset import NBRDatasetBase
from src.models.core import IRecommenderNextTs


FEATURE_COLUMNS = [
    "candidate_from_history",
    "candidate_from_global_top",
    "user_baskets_count",
    "user_avg_basket_size",
    "user_days_since_last_basket",
    "item_global_count",
    "item_global_rank_pct",
    "ui_count_total",
    "ui_repeat_rate",
    "ui_in_last_basket",
    "ui_count_last_3",
    "ui_count_last_5",
    "ui_days_since_last_purchase",
    "ui_mean_gap_days",
    "ui_recency_over_gap",
]


TRAIN_CACHE = {}


def _days_between(later_ts: pd.Timestamp, earlier_ts: pd.Timestamp) -> float:
    return float((later_ts - earlier_ts) / pd.Timedelta(days=1))


def _safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


class LGBMRankerRecommender(IRecommenderNextTs):
    def __init__(
        self,
        global_top_k: int = 100,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self.global_top_k = global_top_k
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._model = None
        self._num_users = None
        self._num_items = None
        self._train_df = None
        self._user_histories = None
        self._global_top_items = None
        self._global_item_count = None
        self._global_item_rank_pct = None

    def fit(self, dataset: NBRDatasetBase):
        cache_key = self._make_cache_key(dataset)
        if cache_key not in TRAIN_CACHE:
            TRAIN_CACHE[cache_key] = self._build_training_cache(dataset)

        cached = TRAIN_CACHE[cache_key]
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        self._train_df = cached["train_df"]
        self._user_histories = cached["user_histories"]
        self._global_top_items = cached["global_top_items"]
        self._global_item_count = cached["global_item_count"]
        self._global_item_rank_pct = cached["global_item_rank_pct"]

        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="None",
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
        )
        self._model.fit(
            cached["x_train"],
            cached["y_train"],
            group=cached["group_train"],
        )
        return self

    def predict(self, user_ids, user_next_basket_ts: pd.DataFrame, topk=None):
        if self._model is None:
            raise RuntimeError("fit the model first")

        if topk is None:
            topk = self._num_items

        next_ts_by_user = dict(
            zip(
                user_next_basket_ts["user_id"].astype(int),
                pd.to_datetime(user_next_basket_ts["next_basket_ts"]),
            )
        )

        rows = []
        query_row_ids = []
        item_ids = []

        for row_idx, user_id in enumerate(user_ids):
            user_id = int(user_id)
            target_ts = pd.Timestamp(next_ts_by_user[user_id])
            history_df = self._user_histories.get(user_id)
            if history_df is None:
                history_df = self._empty_history_df()

            query_rows = self._build_query_rows(
                user_id=user_id,
                history_df=history_df,
                target_ts=target_ts,
                target_items=None,
                require_positive=False,
                global_top_items=self._global_top_items,
                global_item_count=self._global_item_count,
                global_item_rank_pct=self._global_item_rank_pct,
            )
            if len(query_rows) == 0:
                continue

            rows.extend(query_rows)
            query_row_ids.extend([row_idx] * len(query_rows))
            item_ids.extend([int(row["item_id"]) for row in query_rows])

        if len(rows) == 0:
            return sps.csr_matrix((len(user_ids), self._num_items), dtype=np.float32)

        pred_df = pd.DataFrame(rows)
        x_pred = pred_df.loc[:, FEATURE_COLUMNS].astype(np.float32)
        pred_scores = self._model.predict(x_pred)

        score_df = pd.DataFrame(
            {
                "row_idx": query_row_ids,
                "item_id": item_ids,
                "score": pred_scores,
            }
        )
        score_df = (
            score_df.sort_values(["row_idx", "score"], ascending=[True, False])
            .groupby("row_idx", as_index=False, sort=False)
            .head(topk)
        )

        pred_matrix = sps.csr_matrix(
            (
                score_df["score"].astype(np.float32),
                (score_df["row_idx"].astype(int), score_df["item_id"].astype(int)),
            ),
            shape=(len(user_ids), self._num_items),
            dtype=np.float32,
        )
        return pred_matrix

    def _make_cache_key(self, dataset: NBRDatasetBase):
        max_ts = pd.Timestamp(dataset.train_df["timestamp"].max())
        return (
            getattr(dataset, "dataset_dir", "unknown"),
            dataset.num_users,
            dataset.num_items,
            len(dataset.train_df),
            str(max_ts),
            int(self.global_top_k),
        )

    def _build_training_cache(self, dataset: NBRDatasetBase):
        train_df = dataset.train_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        user_histories = {
            int(user_id): user_df.reset_index(drop=True).copy()
            for user_id, user_df in train_df.groupby("user_id", sort=False)
        }

        exploded = train_df.loc[:, ["basket"]].explode("basket", ignore_index=True)
        exploded = exploded.rename(columns={"basket": "item_id"})
        exploded["item_id"] = exploded["item_id"].astype(int)
        global_item_count = exploded.groupby("item_id").size().to_dict()

        global_item_count_sorted = sorted(
            global_item_count.items(),
            key=lambda x: (-int(x[1]), int(x[0])),
        )
        global_top_items = [int(item_id) for item_id, _ in global_item_count_sorted[: self.global_top_k]]

        num_ranked_items = max(len(global_item_count_sorted), 1)
        global_item_rank_pct = {}
        for rank, (item_id, _) in enumerate(global_item_count_sorted, start=1):
            global_item_rank_pct[int(item_id)] = 1.0 - ((rank - 1) / max(num_ranked_items - 1, 1))

        rows = []
        group_train = []
        for user_id, user_df in user_histories.items():
            if len(user_df) < 2:
                continue

            history_df = user_df.iloc[:-1].reset_index(drop=True)
            target_row = user_df.iloc[-1]
            target_ts = pd.Timestamp(target_row["timestamp"])
            target_items = list(map(int, target_row["basket"]))

            query_rows = self._build_query_rows(
                user_id=user_id,
                history_df=history_df,
                target_ts=target_ts,
                target_items=target_items,
                require_positive=True,
                global_top_items=global_top_items,
                global_item_count=global_item_count,
                global_item_rank_pct=global_item_rank_pct,
            )
            if len(query_rows) == 0:
                continue

            rows.extend(query_rows)
            group_train.append(len(query_rows))

        train_rows = pd.DataFrame(rows)
        x_train = train_rows.loc[:, FEATURE_COLUMNS].astype(np.float32)
        y_train = train_rows["label"].astype(np.int32).to_numpy()

        return {
            "train_df": train_df,
            "user_histories": user_histories,
            "global_top_items": global_top_items,
            "global_item_count": global_item_count,
            "global_item_rank_pct": global_item_rank_pct,
            "x_train": x_train,
            "y_train": y_train,
            "group_train": group_train,
        }

    def _build_query_rows(
            self,
            user_id: int,
            history_df: pd.DataFrame,
            target_ts: pd.Timestamp,
            target_items: list[int] | None,
            require_positive: bool,
            global_top_items: list[int],
            global_item_count: dict[int, int],
            global_item_rank_pct: dict[int, float],
    ):
        history_items = set()
        item_count = defaultdict(int)
        item_ts = defaultdict(list)
        item_count_last_3 = defaultdict(int)
        item_count_last_5 = defaultdict(int)

        user_baskets = history_df["basket"].tolist()
        history_timestamps = pd.to_datetime(history_df["timestamp"]).tolist()
        num_history_baskets = len(user_baskets)

        last_3_start = max(0, num_history_baskets - 3)
        last_5_start = max(0, num_history_baskets - 5)

        for basket_idx, (basket, basket_ts) in enumerate(zip(user_baskets, history_timestamps)):
            basket_unique = set(map(int, basket))
            history_items.update(basket_unique)
            for item_id in basket_unique:
                item_count[item_id] += 1
                item_ts[item_id].append(pd.Timestamp(basket_ts))
                if basket_idx >= last_3_start:
                    item_count_last_3[item_id] += 1
                if basket_idx >= last_5_start:
                    item_count_last_5[item_id] += 1

        candidates = list(sorted(history_items))
        history_candidate_set = set(candidates)
        for item_id in global_top_items:
            if item_id not in history_candidate_set:
                candidates.append(int(item_id))

        if len(candidates) == 0:
            return []

        target_set = set() if target_items is None else set(map(int, target_items))
        if require_positive and not any(item_id in target_set for item_id in candidates):
            return []

        user_baskets_count = int(num_history_baskets)
        user_avg_basket_size = _safe_mean([len(basket) for basket in user_baskets])
        if user_baskets_count > 0:
            user_days_since_last_basket = _days_between(target_ts, pd.Timestamp(history_timestamps[-1]))
        else:
            user_days_since_last_basket = -1.0

        rows = []
        for item_id in candidates:
            purchase_ts = item_ts.get(item_id, [])
            ui_count_total = int(item_count.get(item_id, 0))
            ui_count_last_3 = int(item_count_last_3.get(item_id, 0))
            ui_count_last_5 = int(item_count_last_5.get(item_id, 0))
            ui_in_last_basket = int(
                user_baskets_count > 0 and item_id in set(map(int, user_baskets[-1]))
            )

            if len(purchase_ts) > 0:
                ui_days_since_last_purchase = _days_between(target_ts, purchase_ts[-1])
            else:
                ui_days_since_last_purchase = -1.0

            if len(purchase_ts) >= 2:
                gaps = [
                    _days_between(purchase_ts[i], purchase_ts[i - 1])
                    for i in range(1, len(purchase_ts))
                ]
                ui_mean_gap_days = _safe_mean(gaps)
            else:
                ui_mean_gap_days = 0.0

            if ui_days_since_last_purchase >= 0 and ui_mean_gap_days > 0:
                ui_recency_over_gap = ui_days_since_last_purchase / ui_mean_gap_days
            else:
                ui_recency_over_gap = 0.0

            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "label": int(item_id in target_set),
                    "candidate_from_history": int(item_id in history_candidate_set),
                    "candidate_from_global_top": int(item_id in global_top_items),
                    "user_baskets_count": user_baskets_count,
                    "user_avg_basket_size": float(user_avg_basket_size),
                    "user_days_since_last_basket": float(user_days_since_last_basket),
                    "item_global_count": int(global_item_count.get(int(item_id), 0)),
                    "item_global_rank_pct": float(global_item_rank_pct.get(int(item_id), 0.0)),
                    "ui_count_total": ui_count_total,
                    "ui_repeat_rate": float(ui_count_total / max(user_baskets_count, 1)),
                    "ui_in_last_basket": ui_in_last_basket,
                    "ui_count_last_3": ui_count_last_3,
                    "ui_count_last_5": ui_count_last_5,
                    "ui_days_since_last_purchase": float(ui_days_since_last_purchase),
                    "ui_mean_gap_days": float(ui_mean_gap_days),
                    "ui_recency_over_gap": float(ui_recency_over_gap),
                }
            )

        return rows

    @staticmethod
    def _empty_history_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": pd.Series(dtype=int),
                "basket": pd.Series(dtype=object),
                "timestamp": pd.Series(dtype="datetime64[ns]"),
            }
        )

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        return {
            "global_top_k": 100,
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [10, 20, 50]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.85, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.85, 1.0]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 1e-3, 1e-1]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0.0, 1e-3, 1e-1]),
            "random_state": 42,
            "n_jobs": -1,
        }