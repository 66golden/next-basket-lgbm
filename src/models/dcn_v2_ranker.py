from __future__ import annotations

from collections import defaultdict

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import NBRDatasetBase
from src.models.core import IRecommenderNextTs
from src.models.lgbm_ranker import (
    FEATURE_COLUMNS,
    LGBMRankerRecommender,
)


DCN_TRAIN_CACHE = {}


class CrossLayerV2(nn.Module):
    """
    DCN v2 cross layer:
        x_{l+1} = x_0 * W_l(x_l) + b_l + x_l

    здесь W_l — обычный Linear(input_dim -> input_dim).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossLayerV2(input_dim=input_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DCNV2Model(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_dense_features: int,
        user_emb_dim: int = 16,
        item_emb_dim: int = 32,
        cross_layers: int = 2,
        deep_hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)

        input_dim = num_dense_features + user_emb_dim + item_emb_dim

        self.cross_network = CrossNetworkV2(
            input_dim=input_dim,
            num_layers=cross_layers,
        )

        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.deep_network = nn.Sequential(*deep_layers)

        self.output_layer = nn.Linear(input_dim + prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x = torch.cat([dense_features, user_emb, item_emb], dim=1)

        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        out = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output_layer(out).squeeze(1)
        return logits


class DCNV2RankerRecommender(LGBMRankerRecommender, IRecommenderNextTs):
    """
    нейросетевой ranker на тех же candidate rows и feature columns, что lgbm_ranker.

    отличия от lgbm:
    - числовые признаки проходят через dense part;
    - user_id и item_id проходят через trainable embeddings;
    - DCN v2 cross-network явно моделирует feature interactions.
    """

    def __init__(
        self,
        global_top_k: int = 100,
        user_emb_dim: int = 16,
        item_emb_dim: int = 32,
        cross_layers: int = 2,
        deep_hidden_dim_1: int = 128,
        deep_hidden_dim_2: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 8192,
        epochs: int = 5,
        negative_sample_ratio: int = 0,
        random_state: int = 42,
        device: str = "auto",
        n_jobs: int = -1,
    ):
        super().__init__(global_top_k=global_top_k)

        self.user_emb_dim = user_emb_dim
        self.item_emb_dim = item_emb_dim
        self.cross_layers = cross_layers
        self.deep_hidden_dim_1 = deep_hidden_dim_1
        self.deep_hidden_dim_2 = deep_hidden_dim_2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.negative_sample_ratio = negative_sample_ratio
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs

        self._torch_model = None
        self._feature_mean = None
        self._feature_std = None
        self._device = None

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, dataset: NBRDatasetBase):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        cache_key = self._make_cache_key(dataset)
        cache_key = ("dcn_v2", cache_key, int(self.negative_sample_ratio))

        if cache_key not in DCN_TRAIN_CACHE:
            DCN_TRAIN_CACHE[cache_key] = self._build_training_cache(dataset)

        cached = DCN_TRAIN_CACHE[cache_key]

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        self._train_df = cached["train_df"]
        self._user_histories = cached["user_histories"]
        self._global_top_items = cached["global_top_items"]
        self._global_item_count = cached["global_item_count"]
        self._global_item_rank_pct = cached["global_item_rank_pct"]
        self._global_item_ts = cached["global_item_ts"]
        self._global_item_gap_cumsum = cached["global_item_gap_cumsum"]

        x_train = cached["x_train"].astype(np.float32)
        y_train = cached["y_train"].astype(np.float32)
        user_ids = cached["user_ids"].astype(np.int64)
        item_ids = cached["item_ids"].astype(np.int64)

        self._feature_mean = x_train.mean(axis=0).astype(np.float32)
        self._feature_std = x_train.std(axis=0).astype(np.float32)
        self._feature_std[self._feature_std < 1e-6] = 1.0

        x_train = (x_train - self._feature_mean) / self._feature_std

        self._device = self._resolve_device()

        self._torch_model = DCNV2Model(
            num_users=self._num_users,
            num_items=self._num_items,
            num_dense_features=len(FEATURE_COLUMNS),
            user_emb_dim=self.user_emb_dim,
            item_emb_dim=self.item_emb_dim,
            cross_layers=self.cross_layers,
            deep_hidden_dims=(self.deep_hidden_dim_1, self.deep_hidden_dim_2),
            dropout=self.dropout,
        ).to(self._device)

        x_tensor = torch.from_numpy(x_train)
        y_tensor = torch.from_numpy(y_train)
        user_tensor = torch.from_numpy(user_ids)
        item_tensor = torch.from_numpy(item_ids)

        train_dataset = TensorDataset(user_tensor, item_tensor, x_tensor, y_tensor)

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        pos_count = float(y_train.sum())
        neg_count = float(len(y_train) - pos_count)
        pos_weight_value = neg_count / max(pos_count, 1.0)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(self._device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self._torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self._torch_model.train()
        for _ in range(self.epochs):
            for batch_user_ids, batch_item_ids, batch_x, batch_y in train_loader:
                batch_user_ids = batch_user_ids.to(self._device)
                batch_item_ids = batch_item_ids.to(self._device)
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self._torch_model(batch_user_ids, batch_item_ids, batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, user_ids, user_next_basket_ts: pd.DataFrame, topk=None):
        if self._torch_model is None:
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
        user_id_values = []

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
                global_item_ts=self._global_item_ts,
                global_item_gap_cumsum=self._global_item_gap_cumsum,
            )

            if len(query_rows) == 0:
                continue

            rows.extend(query_rows)
            query_row_ids.extend([row_idx] * len(query_rows))
            item_ids.extend([int(row["item_id"]) for row in query_rows])
            user_id_values.extend([user_id] * len(query_rows))

        if len(rows) == 0:
            return sps.csr_matrix((len(user_ids), self._num_items), dtype=np.float32)

        pred_df = pd.DataFrame(rows)
        x_pred = pred_df.loc[:, FEATURE_COLUMNS].astype(np.float32).to_numpy()
        x_pred = (x_pred - self._feature_mean) / self._feature_std

        pred_scores = self._predict_scores_numpy(
            user_ids=np.asarray(user_id_values, dtype=np.int64),
            item_ids=np.asarray(item_ids, dtype=np.int64),
            x_values=x_pred.astype(np.float32),
        )

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

    def _predict_scores_numpy(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        x_values: np.ndarray,
    ) -> np.ndarray:
        self._torch_model.eval()

        scores = []
        batch_size = max(self.batch_size * 2, 4096)

        with torch.no_grad():
            for start in range(0, len(x_values), batch_size):
                end = min(start + batch_size, len(x_values))

                batch_user_ids = torch.from_numpy(user_ids[start:end]).to(self._device)
                batch_item_ids = torch.from_numpy(item_ids[start:end]).to(self._device)
                batch_x = torch.from_numpy(x_values[start:end]).to(self._device)

                logits = self._torch_model(batch_user_ids, batch_item_ids, batch_x)
                scores.append(logits.detach().cpu().numpy())

        return np.concatenate(scores, axis=0).astype(np.float32)

    def _build_training_cache(self, dataset: NBRDatasetBase):
        train_df = dataset.train_df.sort_values(["user_id", "timestamp"]).reset_index(
            drop=True
        )
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

        global_top_items = [
            int(item_id)
            for item_id, _ in global_item_count_sorted[: self.global_top_k]
        ]

        num_ranked_items = max(len(global_item_count_sorted), 1)
        global_item_rank_pct = {}
        for rank, (item_id, _) in enumerate(global_item_count_sorted, start=1):
            global_item_rank_pct[int(item_id)] = 1.0 - (
                (rank - 1) / max(num_ranked_items - 1, 1)
            )

        global_item_ts, global_item_gap_cumsum = self._build_global_item_time_index(
            train_df
        )

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
                global_item_ts=global_item_ts,
                global_item_gap_cumsum=global_item_gap_cumsum,
            )

            if len(query_rows) == 0:
                continue

            rows.extend(query_rows)
            group_train.append(len(query_rows))

        train_rows = pd.DataFrame(rows)

        if len(train_rows) == 0:
            raise RuntimeError(
                "No train rows were built for DCNV2RankerRecommender. "
                "Check candidate generation and split files."
            )

        train_rows = self._negative_sample_train_rows(train_rows)

        x_train = train_rows.loc[:, FEATURE_COLUMNS].astype(np.float32).to_numpy()
        y_train = train_rows["label"].astype(np.int32).to_numpy()
        user_ids = train_rows["user_id"].astype(np.int64).to_numpy()
        item_ids = train_rows["item_id"].astype(np.int64).to_numpy()

        return {
            "train_df": train_df,
            "user_histories": user_histories,
            "global_top_items": global_top_items,
            "global_item_count": global_item_count,
            "global_item_rank_pct": global_item_rank_pct,
            "global_item_ts": global_item_ts,
            "global_item_gap_cumsum": global_item_gap_cumsum,
            "x_train": x_train,
            "y_train": y_train,
            "user_ids": user_ids,
            "item_ids": item_ids,
            "group_train": group_train,
        }

    def _negative_sample_train_rows(self, train_rows: pd.DataFrame) -> pd.DataFrame:
        """
        negative_sample_ratio = 0 означает не сэмплировать negatives.
        если ratio > 0, на каждый positive внутри query берем до ratio negatives.

        это ускоряет dcn, но меняет train distribution.
        для первого честного сравнения лучше оставить 0.
        """

        if self.negative_sample_ratio <= 0:
            return train_rows.reset_index(drop=True)

        rng = np.random.default_rng(self.random_state)

        sampled_parts = []

        for _, query_df in train_rows.groupby("user_id", sort=False):
            pos_df = query_df[query_df["label"] == 1]
            neg_df = query_df[query_df["label"] == 0]

            if len(pos_df) == 0:
                continue

            max_neg = min(len(neg_df), len(pos_df) * self.negative_sample_ratio)
            if max_neg > 0:
                neg_idx = rng.choice(neg_df.index.to_numpy(), size=max_neg, replace=False)
                sampled_parts.append(pd.concat([pos_df, neg_df.loc[neg_idx]], axis=0))
            else:
                sampled_parts.append(pos_df)

        if len(sampled_parts) == 0:
            return train_rows.reset_index(drop=True)

        sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=self.random_state)
        return sampled.reset_index(drop=True)

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        return {
            "global_top_k": trial.suggest_categorical("global_top_k", [100, 200]),
            "user_emb_dim": trial.suggest_categorical("user_emb_dim", [8, 16, 32]),
            "item_emb_dim": trial.suggest_categorical("item_emb_dim", [16, 32, 64]),
            "cross_layers": trial.suggest_categorical("cross_layers", [1, 2, 3]),
            "deep_hidden_dim_1": trial.suggest_categorical(
                "deep_hidden_dim_1", [64, 128, 256]
            ),
            "deep_hidden_dim_2": trial.suggest_categorical(
                "deep_hidden_dim_2", [32, 64, 128]
            ),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [3e-4, 1e-3, 3e-3]
            ),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0.0, 1e-6, 1e-5]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [4096, 8192, 16384]
            ),
            "epochs": trial.suggest_categorical("epochs", [3, 5, 8]),
            "negative_sample_ratio": trial.suggest_categorical(
                "negative_sample_ratio", [0]
            ),
            "random_state": 42,
            "device": "auto",
            "n_jobs": -1,
        }