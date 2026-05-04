from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps
import torch
from torch import nn
from torch.nn import functional as F
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
    DCN v2 cross layer.

    formula:
        x_{l+1} = x_0 * W_l(x_l) + x_l

    где:
    - x_0 — исходный input;
    - x_l — текущее представление;
    - W_l — обучаемый linear layer.
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
        user_emb_dim: int = 8,
        item_emb_dim: int = 32,
        cross_layers: int = 2,
        deep_hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.user_emb_dim = int(user_emb_dim)
        self.item_emb_dim = int(item_emb_dim)

        if self.user_emb_dim > 0:
            self.user_embedding = nn.Embedding(num_users, self.user_emb_dim)
        else:
            self.user_embedding = None

        if self.item_emb_dim > 0:
            self.item_embedding = nn.Embedding(num_items, self.item_emb_dim)
        else:
            self.item_embedding = None

        input_dim = num_dense_features + self.user_emb_dim + self.item_emb_dim

        if input_dim <= 0:
            raise ValueError("DCNV2Model input_dim must be positive")

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
        if self.user_embedding is not None:
            nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)

        if self.item_embedding is not None:
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
        parts = [dense_features]

        if self.user_embedding is not None:
            parts.append(self.user_embedding(user_ids))

        if self.item_embedding is not None:
            parts.append(self.item_embedding(item_ids))

        x = torch.cat(parts, dim=1)

        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        out = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output_layer(out).squeeze(1)

        return logits


class DCNV2RankerRecommender(LGBMRankerRecommender, IRecommenderNextTs):
    """
    DCN v2 ranker.

    Использует:
    - тот же candidate generation, что lgbm_ranker;
    - те же handcrafted numerical features;
    - обучаемые item_id embeddings;
    - опциональные user_id embeddings;
    - DCN v2 cross network;
    - BCE loss или BPR / RankNet-style pairwise loss.

    Главная идея:
    сравнить lgbm и нейросетевой ranker в одинаковой постановке:
        same split,
        same candidates,
        same metrics,
        same handcrafted features,
        но dcn дополнительно учит embeddings и feature interactions.
    """

    def __init__(
        self,
        global_top_k: int = 100,
        user_emb_dim: int = 8,
        item_emb_dim: int = 32,
        cross_layers: int = 2,
        deep_hidden_dim_1: int = 64,
        deep_hidden_dim_2: int = 32,
        dropout: float = 0.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 8192,
        epochs: int = 3,
        loss_type: str = "bpr",
        negative_sample_ratio: int = 0,
        bpr_negatives_per_positive: int = 10,
        max_prefix_queries_per_user: int = 1,
        random_state: int = 42,
        device: str = "auto",
        verbose_training: bool = False,
        n_jobs: int = -1,
    ):
        super().__init__(global_top_k=global_top_k)

        if loss_type not in {"bce", "bpr"}:
            raise ValueError("loss_type must be either 'bce' or 'bpr'")

        self.user_emb_dim = int(user_emb_dim)
        self.item_emb_dim = int(item_emb_dim)
        self.cross_layers = int(cross_layers)
        self.deep_hidden_dim_1 = int(deep_hidden_dim_1)
        self.deep_hidden_dim_2 = int(deep_hidden_dim_2)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.loss_type = loss_type
        self.negative_sample_ratio = int(negative_sample_ratio)
        self.bpr_negatives_per_positive = int(bpr_negatives_per_positive)
        self.max_prefix_queries_per_user = int(max_prefix_queries_per_user)
        self.random_state = int(random_state)
        self.device = device
        self.verbose_training = bool(verbose_training)
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

        cache_key = (
            "dcn_v2_features",
            self._make_cache_key(dataset),
            int(self.max_prefix_queries_per_user),
        )

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

        train_rows = cached["train_rows"].copy()

        if self.loss_type == "bce":
            train_rows = self._negative_sample_train_rows(train_rows)

        x_train = train_rows.loc[:, FEATURE_COLUMNS].astype(np.float32).to_numpy()
        y_train = train_rows["label"].astype(np.float32).to_numpy()
        user_ids = train_rows["user_id"].astype(np.int64).to_numpy()
        item_ids = train_rows["item_id"].astype(np.int64).to_numpy()

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

        optimizer = torch.optim.AdamW(
            self._torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.loss_type == "bce":
            self._fit_bce(
                user_ids=user_ids,
                item_ids=item_ids,
                x_train=x_train,
                y_train=y_train,
                optimizer=optimizer,
            )
        else:
            self._fit_bpr(
                train_rows=train_rows,
                x_train=x_train,
                optimizer=optimizer,
            )

        return self

    def _fit_bce(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        optimizer: torch.optim.Optimizer,
    ):
        x_tensor = torch.from_numpy(x_train.astype(np.float32))
        y_tensor = torch.from_numpy(y_train.astype(np.float32))
        user_tensor = torch.from_numpy(user_ids.astype(np.int64))
        item_tensor = torch.from_numpy(item_ids.astype(np.int64))

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

        pos_weight = torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=self._device,
        )

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._torch_model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_batches = 0

            for batch_user_ids, batch_item_ids, batch_x, batch_y in train_loader:
                batch_user_ids = batch_user_ids.to(self._device)
                batch_item_ids = batch_item_ids.to(self._device)
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()

                logits = self._torch_model(
                    user_ids=batch_user_ids,
                    item_ids=batch_item_ids,
                    dense_features=batch_x,
                )

                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_batches += 1

            if self.verbose_training:
                avg_loss = total_loss / max(total_batches, 1)
                print(f"DCN BCE epoch={epoch + 1}/{self.epochs}, loss={avg_loss:.6f}")

    def _fit_bpr(
        self,
        train_rows: pd.DataFrame,
        x_train: np.ndarray,
        optimizer: torch.optim.Optimizer,
    ):
        pair_arrays = self._build_bpr_pair_arrays(
            train_rows=train_rows,
            x_train=x_train,
        )

        (
            pair_user_ids,
            pos_item_ids,
            neg_item_ids,
            pos_x,
            neg_x,
        ) = pair_arrays

        pair_dataset = TensorDataset(
            torch.from_numpy(pair_user_ids.astype(np.int64)),
            torch.from_numpy(pos_item_ids.astype(np.int64)),
            torch.from_numpy(neg_item_ids.astype(np.int64)),
            torch.from_numpy(pos_x.astype(np.float32)),
            torch.from_numpy(neg_x.astype(np.float32)),
        )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        pair_loader = DataLoader(
            pair_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        self._torch_model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_batches = 0

            for (
                batch_user_ids,
                batch_pos_item_ids,
                batch_neg_item_ids,
                batch_pos_x,
                batch_neg_x,
            ) in pair_loader:
                batch_user_ids = batch_user_ids.to(self._device)
                batch_pos_item_ids = batch_pos_item_ids.to(self._device)
                batch_neg_item_ids = batch_neg_item_ids.to(self._device)
                batch_pos_x = batch_pos_x.to(self._device)
                batch_neg_x = batch_neg_x.to(self._device)

                optimizer.zero_grad()

                pos_scores = self._torch_model(
                    user_ids=batch_user_ids,
                    item_ids=batch_pos_item_ids,
                    dense_features=batch_pos_x,
                )

                neg_scores = self._torch_model(
                    user_ids=batch_user_ids,
                    item_ids=batch_neg_item_ids,
                    dense_features=batch_neg_x,
                )

                # RankNet / BPR-style loss:
                # хотим pos_score > neg_score
                loss = F.softplus(neg_scores - pos_scores).mean()

                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_batches += 1

            if self.verbose_training:
                avg_loss = total_loss / max(total_batches, 1)
                print(f"DCN BPR epoch={epoch + 1}/{self.epochs}, loss={avg_loss:.6f}")

    def _build_bpr_pair_arrays(
        self,
        train_rows: pd.DataFrame,
        x_train: np.ndarray,
    ):
        rng = np.random.default_rng(self.random_state)

        pair_user_ids = []
        pos_item_ids = []
        neg_item_ids = []
        pos_x_rows = []
        neg_x_rows = []

        for _, query_df in train_rows.groupby("query_id", sort=False):
            pos_indices = query_df.index[query_df["label"] == 1].to_numpy()
            neg_indices = query_df.index[query_df["label"] == 0].to_numpy()

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            for pos_idx in pos_indices:
                sample_size = min(self.bpr_negatives_per_positive, len(neg_indices))
                sampled_neg_indices = rng.choice(
                    neg_indices,
                    size=sample_size,
                    replace=False,
                )

                for neg_idx in sampled_neg_indices:
                    pair_user_ids.append(int(train_rows.at[pos_idx, "user_id"]))
                    pos_item_ids.append(int(train_rows.at[pos_idx, "item_id"]))
                    neg_item_ids.append(int(train_rows.at[neg_idx, "item_id"]))
                    pos_x_rows.append(x_train[pos_idx])
                    neg_x_rows.append(x_train[neg_idx])

        if len(pair_user_ids) == 0:
            raise RuntimeError(
                "No BPR pairs were built. "
                "Check candidate generation or switch loss_type='bce'."
            )

        return (
            np.asarray(pair_user_ids, dtype=np.int64),
            np.asarray(pos_item_ids, dtype=np.int64),
            np.asarray(neg_item_ids, dtype=np.int64),
            np.asarray(pos_x_rows, dtype=np.float32),
            np.asarray(neg_x_rows, dtype=np.float32),
        )

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
            return sps.csr_matrix(
                (len(user_ids), self._num_items),
                dtype=np.float32,
            )

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

                logits = self._torch_model(
                    user_ids=batch_user_ids,
                    item_ids=batch_item_ids,
                    dense_features=batch_x,
                )

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
        query_id = 0

        for user_id, user_df in user_histories.items():
            target_indices = self._get_prefix_target_indices(len(user_df))

            for target_idx in target_indices:
                history_df = user_df.iloc[:target_idx].reset_index(drop=True)
                target_row = user_df.iloc[target_idx]

                target_ts = pd.Timestamp(target_row["timestamp"])
                target_items = list(map(int, target_row["basket"]))

                query_rows = self._build_query_rows(
                    user_id=int(user_id),
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

                for row in query_rows:
                    row["query_id"] = query_id

                rows.extend(query_rows)
                query_id += 1

        train_rows = pd.DataFrame(rows)

        if len(train_rows) == 0:
            raise RuntimeError(
                "No train rows were built for DCNV2RankerRecommender. "
                "Check candidate generation and split files."
            )

        train_rows = train_rows.reset_index(drop=True)

        return {
            "train_df": train_df,
            "user_histories": user_histories,
            "global_top_items": global_top_items,
            "global_item_count": global_item_count,
            "global_item_rank_pct": global_item_rank_pct,
            "global_item_ts": global_item_ts,
            "global_item_gap_cumsum": global_item_gap_cumsum,
            "train_rows": train_rows,
        }

    def _get_prefix_target_indices(self, user_history_len: int) -> list[int]:
        """
        Возвращает индексы target basket внутри train history пользователя.

        Старый режим:
            max_prefix_queries_per_user = 1
            берем только последнюю train-корзину как target.

        Новый ограниченный prefix-train:
            max_prefix_queries_per_user = 3 или 5
            берем несколько последних target-корзин пользователя.
        """

        if user_history_len < 2:
            return []

        all_target_indices = list(range(1, user_history_len))

        if self.max_prefix_queries_per_user <= 1:
            return [all_target_indices[-1]]

        return all_target_indices[-self.max_prefix_queries_per_user :]

    def _negative_sample_train_rows(self, train_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Для BCE можно оставить все negatives или сэмплировать negatives внутри query.

        negative_sample_ratio = 0:
            не сэмплируем, используем все candidate rows.

        negative_sample_ratio > 0:
            на каждый positive внутри query берем до ratio negatives.
        """

        if self.negative_sample_ratio <= 0:
            return train_rows.reset_index(drop=True)

        rng = np.random.default_rng(self.random_state)
        sampled_parts = []

        for _, query_df in train_rows.groupby("query_id", sort=False):
            pos_df = query_df[query_df["label"] == 1]
            neg_df = query_df[query_df["label"] == 0]

            if len(pos_df) == 0:
                continue

            max_neg = min(len(neg_df), len(pos_df) * self.negative_sample_ratio)

            if max_neg > 0:
                neg_idx = rng.choice(
                    neg_df.index.to_numpy(),
                    size=max_neg,
                    replace=False,
                )
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
            "global_top_k": trial.suggest_categorical("global_top_k", [100, 150, 200]),

            # user embeddings могут переобучаться при одном target basket на пользователя,
            # поэтому обязательно пробуем 0.
            "user_emb_dim": trial.suggest_categorical("user_emb_dim", [0, 4, 8]),

            # item embeddings оставляем как главный нейросетевой сигнал.
            "item_emb_dim": trial.suggest_categorical("item_emb_dim", [16, 32, 64]),

            # dcn лучше начинать с маленькой модели.
            "cross_layers": trial.suggest_categorical("cross_layers", [1, 2]),
            "deep_hidden_dim_1": trial.suggest_categorical(
                "deep_hidden_dim_1", [32, 64, 128]
            ),
            "deep_hidden_dim_2": trial.suggest_categorical(
                "deep_hidden_dim_2", [16, 32, 64]
            ),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1]),

            # маленький lr оказался важен для tafeng, но 1e-3 тоже стоит проверить.
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1e-4, 3e-4, 1e-3]
            ),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0.0, 1e-6]
            ),

            "batch_size": trial.suggest_categorical("batch_size", [4096, 8192]),
            "epochs": trial.suggest_categorical("epochs", [2, 3, 5]),

            # главный новый параметр.
            "loss_type": trial.suggest_categorical("loss_type", ["bpr", "bce"]),

            # используется только для BCE.
            "negative_sample_ratio": trial.suggest_categorical(
                "negative_sample_ratio", [0]
            ),

            # используется только для BPR.
            "bpr_negatives_per_positive": trial.suggest_categorical(
                "bpr_negatives_per_positive", [5, 10, 20]
            ),

            # ограниченный prefix-train.
            # 1 = старый режим, 3 = несколько последних train targets.
            "max_prefix_queries_per_user": trial.suggest_categorical(
                "max_prefix_queries_per_user", [1, 3]
            ),

            "random_state": 42,
            "device": "auto",
            "verbose_training": False,
            "n_jobs": -1,
        }