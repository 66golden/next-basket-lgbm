import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from src.dataset import DATASETS
from src.models.lgbm_ranker import (
    FEATURE_COLUMNS,
    TRAIN_CACHE,
    LGBMRankerRecommender,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument(
        "--explain-split",
        type=str,
        default="validate",
        choices=["train", "validate", "test"],
        help="На каком split считать SHAP. По умолчанию validate.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Сколько candidate rows брать для SHAP. Если строк больше, будет сэмплирование.",
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.5,
        help="Доля positive rows в SHAP-сэмпле, если positives есть.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Куда сохранить аналитику. По умолчанию results/analysis/{dataset}_lgbm",
    )
    parser.add_argument(
        "--use-best-db",
        action="store_true",
        help="Взять лучшие параметры из results/{dataset}_lgbm_ranker.db",
    )
    return parser.parse_args()


def load_best_params(dataset_name: str) -> dict:
    db_path = Path("results") / f"{dataset_name}_lgbm_ranker.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Не найден файл optuna db: {db_path}")

    study_name = f"{dataset_name}_lgbm_ranker"
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_params


def build_rows_for_split(
    model: LGBMRankerRecommender,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for query_id, row in split_df.reset_index(drop=True).iterrows():
        user_id = int(row["user_id"])
        target_ts = pd.Timestamp(row["timestamp"])
        target_items = list(map(int, row["basket"]))

        history_df = model._user_histories.get(user_id)
        if history_df is None:
            history_df = model._empty_history_df()

        query_rows = model._build_query_rows(
            user_id=user_id,
            history_df=history_df,
            target_ts=target_ts,
            target_items=target_items,
            require_positive=False,
            global_top_items=model._global_top_items,
            global_item_count=model._global_item_count,
            global_item_rank_pct=model._global_item_rank_pct,
        )

        if len(query_rows) == 0:
            continue

        for qr in query_rows:
            qr["query_id"] = int(query_id)
            qr["target_ts"] = target_ts
        rows.extend(query_rows)

    if len(rows) == 0:
        raise RuntimeError("Не удалось построить candidate rows для выбранного split.")

    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, sample_size: int, positive_ratio: float, random_state: int = 42):
    if len(df) <= sample_size:
        return df.copy().reset_index(drop=True)

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    if len(pos_df) == 0 or len(neg_df) == 0:
        return df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    pos_n = min(len(pos_df), int(sample_size * positive_ratio))
    neg_n = min(len(neg_df), sample_size - pos_n)

    sampled = pd.concat(
        [
            pos_df.sample(pos_n, random_state=random_state),
            neg_df.sample(neg_n, random_state=random_state),
        ],
        axis=0,
        ignore_index=True,
    )

    if len(sampled) < sample_size:
        remaining = df.drop(sampled.index, errors="ignore")
        need = min(sample_size - len(sampled), len(remaining))
        if need > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(need, random_state=random_state)],
                axis=0,
                ignore_index=True,
            )

    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def make_barplot(series: pd.Series, title: str, xlabel: str, path: Path):
    s = series.sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(s.index, s.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    args = parse_args()

    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(verbose=True)
    dataset.load_split()

    if args.use_best_db:
        best_params = load_best_params(args.dataset)
    else:
        best_params = {
            "global_top_k": 100,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    best_params = {
        "global_top_k": 100,
        "random_state": 42,
        "n_jobs": -1,
        **best_params,
    }

    model = LGBMRankerRecommender(**best_params)
    model.fit(dataset)

    booster: lgb.Booster = model._model.booster_
    cache_key = model._make_cache_key(dataset)
    train_cache = TRAIN_CACHE[cache_key]

    if args.explain_split == "train":
        explain_df = train_cache["x_train"].copy()
        explain_df["label"] = train_cache["y_train"]
    elif args.explain_split == "validate":
        explain_df = build_rows_for_split(model, dataset.val_df)
    else:
        explain_df = build_rows_for_split(model, dataset.test_df)

    explain_df = stratified_sample(
        explain_df,
        sample_size=args.sample_size,
        positive_ratio=args.positive_ratio,
        random_state=42,
    )

    x_explain = explain_df.loc[:, FEATURE_COLUMNS].astype(np.float32)
    raw_scores = booster.predict(x_explain)
    shap_values = booster.predict(x_explain, pred_contrib=True)

    shap_matrix = shap_values[:, :-1]
    shap_bias = shap_values[:, -1]

    gain_importance = booster.feature_importance(importance_type="gain")
    split_importance = booster.feature_importance(importance_type="split")

    mean_abs_shap_all = np.abs(shap_matrix).mean(axis=0)

    pos_mask = explain_df["label"].to_numpy() == 1
    if pos_mask.sum() > 0:
        mean_abs_shap_pos = np.abs(shap_matrix[pos_mask]).mean(axis=0)
        mean_shap_pos = shap_matrix[pos_mask].mean(axis=0)
    else:
        mean_abs_shap_pos = np.full(len(FEATURE_COLUMNS), np.nan)
        mean_shap_pos = np.full(len(FEATURE_COLUMNS), np.nan)

    mean_feature_value = x_explain.mean(axis=0).to_numpy()

    summary = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_gain": gain_importance,
            "importance_split": split_importance,
            "mean_abs_shap_all": mean_abs_shap_all,
            "mean_abs_shap_positive": mean_abs_shap_pos,
            "mean_shap_positive": mean_shap_pos,
            "mean_feature_value": mean_feature_value,
        }
    )

    summary["gain_pct"] = summary["importance_gain"] / max(summary["importance_gain"].sum(), 1e-12)
    summary["split_pct"] = summary["importance_split"] / max(summary["importance_split"].sum(), 1e-12)
    summary = summary.sort_values("mean_abs_shap_all", ascending=False).reset_index(drop=True)

    if args.output_dir is None:
        output_dir = Path("results") / "analysis" / f"{args.dataset}_lgbm"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / f"{args.dataset}_summary.csv", index=False)

    sample_to_save = explain_df.copy()
    sample_to_save["raw_score"] = raw_scores
    sample_to_save["shap_bias"] = shap_bias
    sample_to_save.to_csv(output_dir / f"{args.dataset}_sample_rows.csv", index=False)

    with open(output_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    meta = {
        "dataset": args.dataset,
        "explain_split": args.explain_split,
        "num_train_rows": int(len(train_cache["x_train"])),
        "num_explained_rows": int(len(explain_df)),
        "positive_rows_explained": int(pos_mask.sum()),
        "negative_rows_explained": int((~pos_mask).sum()),
        "num_features": len(FEATURE_COLUMNS),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    make_barplot(
        summary.set_index("feature")["importance_gain"],
        title=f"{args.dataset}: feature importance (gain)",
        xlabel="gain",
        path=output_dir / f"{args.dataset}_importance_gain.png",
    )

    make_barplot(
        summary.set_index("feature")["mean_abs_shap_all"],
        title=f"{args.dataset}: mean |SHAP| on {args.explain_split}",
        xlabel="mean |SHAP|",
        path=output_dir / f"{args.dataset}_shap_mean_abs_all.png",
    )

    print()
    print("saved to:", output_dir)
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()