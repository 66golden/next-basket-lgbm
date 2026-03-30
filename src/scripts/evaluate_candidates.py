from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd

from src.dataset import DATASETS
from src.features import build_eval_states, SimpleCandidateGenerator


def candidate_recall_at_k(
    true_items: Sequence[int],
    candidate_items: Sequence[int],
    k: int,
) -> float:
    true_set = set(map(int, true_items))
    cand_topk = list(map(int, list(candidate_items)[:k]))

    if len(true_set) == 0:
        return 0.0

    hits = len(true_set.intersection(cand_topk))
    return float(hits / len(true_set))


def candidate_full_recall(
    true_items: Sequence[int],
    candidate_items: Sequence[int],
) -> float:
    true_set = set(map(int, true_items))
    cand_set = set(map(int, candidate_items))

    if len(true_set) == 0:
        return 0.0

    hits = len(true_set.intersection(cand_set))
    return float(hits / len(true_set))


def candidate_hit_rate(
    true_items: Sequence[int],
    candidate_items: Sequence[int],
) -> float:
    true_set = set(map(int, true_items))
    cand_set = set(map(int, candidate_items))

    if len(true_set) == 0:
        return 0.0

    return float(len(true_set.intersection(cand_set)) > 0)


def evaluate_candidate_generator(
    train_df: pd.DataFrame,
    eval_states_df: pd.DataFrame,
    candidate_generator,
    cutoffs: list[int],
) -> dict:
    metrics_acc = {
        "candidate_full_recall": [],
        "candidate_hit_rate": [],
    }
    for k in cutoffs:
        metrics_acc[f"candidate_recall@{k}"] = []

    user_histories = {
        int(user_id): user_df.sort_values("timestamp").reset_index(drop=True).copy()
        for user_id, user_df in train_df.groupby("user_id", sort=False)
    }

    total_queries = len(eval_states_df)

    for idx, (_, row) in enumerate(eval_states_df.iterrows(), start=1):
        if idx == 1 or idx % 1000 == 0 or idx == total_queries:
            print(f"[evaluate_candidates] processed {idx}/{total_queries} queries")

        user_id = int(row["user_id"])
        target_timestamp = pd.Timestamp(row["target_timestamp"])
        target_basket = list(map(int, row["target_basket"]))

        user_history_df = user_histories.get(user_id)
        if user_history_df is None:
            history_before_t = pd.DataFrame(columns=["user_id", "basket", "timestamp"])
        else:
            history_before_t = user_history_df.loc[user_history_df["timestamp"] < target_timestamp].copy()
            history_before_t = history_before_t.sort_values("timestamp").reset_index(drop=True)

        candidates = candidate_generator.generate_for_history(history_before_t)
        candidates = list(map(int, candidates))

        metrics_acc["candidate_full_recall"].append(
            candidate_full_recall(target_basket, candidates)
        )
        metrics_acc["candidate_hit_rate"].append(
            candidate_hit_rate(target_basket, candidates)
        )
        for k in cutoffs:
            metrics_acc[f"candidate_recall@{k}"].append(
                candidate_recall_at_k(target_basket, candidates, k)
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

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset
    dataset_dir_name = args.dataset_dir_name or dataset_name

    dataset_cls = DATASETS[dataset_name]
    data = dataset_cls(dataset_dir_name, verbose=True)
    data.load_split()

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

    candidate_generator = SimpleCandidateGenerator(global_top_m=args.global_top_m)
    candidate_generator.fit(data.train_df)

    print("evaluating validation candidates...")
    val_metrics = evaluate_candidate_generator(
        train_df=data.train_df,
        eval_states_df=val_states,
        candidate_generator=candidate_generator,
        cutoffs=[5, 10, 20, 50, 100],
    )
    print("validation candidate metrics:", val_metrics)

    print("evaluating test candidates...")
    test_metrics = evaluate_candidate_generator(
        train_df=data.train_df,
        eval_states_df=test_states,
        candidate_generator=candidate_generator,
        cutoffs=[5, 10, 20, 50, 100],
    )
    print("test candidate metrics:", test_metrics)


if __name__ == "__main__":
    main()