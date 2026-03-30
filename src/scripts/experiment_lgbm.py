from __future__ import annotations

import argparse
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import DATASETS
from src.features import (
    build_eval_states,
    build_train_prefix_states,
    SimpleCandidateGenerator,
    SimpleFeatureBuilder,
)
from src.models.lgbm_ranker import LGBMRankerConfig, LGBMRankerModel
from src.settings import RESULTS_DIR as RESULTS_DIR_RAW

RESULTS_DIR = Path(RESULTS_DIR_RAW)


def recall_at_k_from_items(true_items: list[int], pred_items: list[int], k: int) -> float:
    true_set = set(map(int, true_items))
    if len(true_set) == 0:
        return 0.0

    pred_topk = list(map(int, pred_items[:k]))
    hits = len(true_set.intersection(pred_topk))
    return float(hits / len(true_set))


def dcg_at_k_from_items(true_items: list[int], pred_items: list[int], k: int) -> float:
    true_set = set(map(int, true_items))
    pred_topk = list(map(int, pred_items[:k]))

    dcg = 0.0
    for rank, item_id in enumerate(pred_topk, start=1):
        if item_id in true_set:
            dcg += 1.0 / log2(rank + 1)
    return float(dcg)


def ndcg_at_k_from_items(true_items: list[int], pred_items: list[int], k: int) -> float:
    true_set = set(map(int, true_items))
    if len(true_set) == 0:
        return 0.0

    dcg = dcg_at_k_from_items(true_items, pred_items, k)

    ideal_hits = min(len(true_set), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_ranker_predictions(
    pred_scores_df: pd.DataFrame,
    eval_states_df: pd.DataFrame,
    cutoffs: list[int],
) -> dict:
    pred_scores_df = pred_scores_df.sort_values(
        ["query_id", "score", "item_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    grouped_pred_items = {
        int(query_id): group["item_id"].astype(int).tolist()
        for query_id, group in pred_scores_df.groupby("query_id", sort=False)
    }

    metrics_acc = {}
    for k in cutoffs:
        metrics_acc[f"recall@{k}"] = []
        metrics_acc[f"ndcg@{k}"] = []

    for _, row in eval_states_df.iterrows():
        query_id = int(row["query_id"])
        true_items = list(map(int, row["target_basket"]))
        pred_items = grouped_pred_items.get(query_id, [])

        for k in cutoffs:
            metrics_acc[f"recall@{k}"].append(
                recall_at_k_from_items(true_items, pred_items, k)
            )
            metrics_acc[f"ndcg@{k}"].append(
                ndcg_at_k_from_items(true_items, pred_items, k)
            )

    result = {
        metric_name: float(np.mean(values)) if len(values) > 0 else 0.0
        for metric_name, values in metrics_acc.items()
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-dir-name", type=str, default=None)

    parser.add_argument("--global-top-m", type=int, default=100)
    parser.add_argument("--max-states-per-user", type=int, default=None)
    parser.add_argument("--max-negatives-per-query", type=int, default=200)
    parser.add_argument("--min-history-baskets", type=int, default=1)

    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)

    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset
    dataset_dir_name = args.dataset_dir_name or dataset_name

    dataset_cls = DATASETS[dataset_name]
    data = dataset_cls(dataset_dir_name, verbose=True)
    data.load_split()

    print("building train/validation/test states...")
    train_states = build_train_prefix_states(
        train_df=data.train_df,
        min_history_baskets=args.min_history_baskets,
        max_states_per_user=args.max_states_per_user,
    )
    val_states = build_eval_states(
        train_df=data.train_df,
        target_df=data.val_df,
        source_name="validation",
    )
    test_states = build_eval_states(
        train_df=data.train_df,
        target_df=data.test_df,
        source_name="test",
    )

    print("fitting simple candidate generator...")
    candidate_generator = SimpleCandidateGenerator(global_top_m=args.global_top_m)
    candidate_generator.fit(data.train_df)

    print("fitting feature builder...")
    feature_builder = SimpleFeatureBuilder(
        global_top_items=candidate_generator.global_top_items_,
        max_negatives_per_query=args.max_negatives_per_query,
        random_state=42,
        verbose=True,
        log_every_n_queries=1000,
    )
    feature_builder.fit(data.train_df)

    print("building train dataset...")
    train_result = feature_builder.build_dataset(
        states_df=train_states,
        candidate_generator=candidate_generator,
        train_mode=True,
    )
    train_df = train_result.query_rows_df
    train_group = train_result.group_sizes
    print("train_df shape:", train_df.shape)

    print("building validation dataset...")
    val_result = feature_builder.build_dataset(
        states_df=val_states,
        candidate_generator=candidate_generator,
        train_mode=False,
    )
    val_df = val_result.query_rows_df
    val_group = val_result.group_sizes
    print("val_df shape:", val_df.shape)

    print("building test dataset...")
    test_result = feature_builder.build_dataset(
        states_df=test_states,
        candidate_generator=candidate_generator,
        train_mode=False,
    )
    test_df = test_result.query_rows_df
    test_group = test_result.group_sizes
    print("test_df shape:", test_df.shape)

    print("training lightgbm ranker...")
    config = LGBMRankerConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=42,
        n_jobs=-1,
    )
    model = LGBMRankerModel(config=config)
    model.fit(
        train_df=train_df,
        train_group=train_group,
        valid_df=val_df,
        valid_group=val_group,
    )

    print("predicting validation...")
    val_pred_df = model.predict_scores(val_df)
    val_metrics = evaluate_ranker_predictions(
        pred_scores_df=val_pred_df,
        eval_states_df=val_states,
        cutoffs=[5, 10],
    )
    print("validation metrics:", val_metrics)

    print("predicting test...")
    test_pred_df = model.predict_scores(test_df)
    test_metrics = evaluate_ranker_predictions(
        pred_scores_df=test_pred_df,
        eval_states_df=test_states,
        cutoffs=[5, 10],
    )
    print("test metrics:", test_metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = args.run_name if args.run_name is not None else f"{dataset_dir_name}_lgbm_ranker"

    pd.DataFrame([val_metrics]).to_csv(RESULTS_DIR / f"{prefix}_valid.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / f"{prefix}_test.csv", index=False)

    print("saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
