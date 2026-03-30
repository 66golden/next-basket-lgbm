from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.ranker_utils import FEATURE_COLUMNS, prepare_lgbm_matrices


@dataclass
class LGBMRankerConfig:
    objective: str = "lambdarank"
    metric: str = "ndcg"
    boosting_type: str = "gbdt"
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1


class LGBMRankerModel:
    def __init__(self, config: LGBMRankerConfig):
        self.config = config
        self.model_: lgb.LGBMRanker | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        train_group: list[int],
        valid_df: pd.DataFrame | None = None,
        valid_group: list[int] | None = None,
    ) -> "LGBMRankerModel":
        x_train, y_train = prepare_lgbm_matrices(train_df)

        model = lgb.LGBMRanker(
            objective=self.config.objective,
            metric=self.config.metric,
            boosting_type=self.config.boosting_type,
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        fit_kwargs = {
            "X": x_train,
            "y": y_train,
            "group": train_group,
        }

        if valid_df is not None and valid_group is not None:
            x_valid, y_valid = prepare_lgbm_matrices(valid_df)
            fit_kwargs["eval_set"] = [(x_valid, y_valid)]
            fit_kwargs["eval_group"] = [valid_group]
            fit_kwargs["eval_at"] = [5, 10]

        model.fit(**fit_kwargs)
        self.model_ = model
        return self

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        x = df.loc[:, FEATURE_COLUMNS].copy()
        for col in x.columns:
            x[col] = x[col].astype(float)

        scores = self.model_.predict(x)

        out = df.loc[:, ["query_id", "user_id", "item_id"]].copy()
        out["score"] = scores.astype(float)
        return out