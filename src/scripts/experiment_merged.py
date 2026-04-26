import argparse
import json
import os
from typing import Dict

import numpy as np
import optuna
import pandas as pd

from src.dataset.merged_all import MergedAllDataset
from src.evaluation import Evaluator
from src.metrics import METRICS
from src.models import MODELS
from src.settings import RESULTS_DIR


def metric_key(metric_name: str, cutoff: int) -> str:
    return f"{metric_name}@{cutoff:03d}"


def _params_already_exist(study, params):
    for tr in study.get_trials(deepcopy=False):
        if tr.params == params:
            return True
    return False


def create_study(prefix: str, model_cls):
    storage = f"sqlite:///{os.path.join(RESULTS_DIR, f'{prefix}.db')}"
    sampler = None
    if hasattr(model_cls, "make_sampler"):
        sampler = model_cls.make_sampler()

    study = optuna.create_study(
        study_name=prefix,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    if hasattr(model_cls, "initial_trials"):
        for params in model_cls.initial_trials():
            if not _params_already_exist(study, params):
                study.enqueue_trial(params)

    return study


def run_merged_experiment(
    model_name: str,
    metric_name: str,
    cutoff: int,
    num_trials: int,
    batch_size: int,
    dataset_names: list[str],
    valid_mode: str = "macro",
    verbose: bool = True,
):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")

    cutoff_list = [5, 10, 15, 20, 50, 100]
    if cutoff not in cutoff_list:
        raise ValueError(f"Unsupported cutoff: {cutoff}")

    model_cls = MODELS[model_name]
    data = MergedAllDataset(dataset_names=dataset_names, verbose=verbose)
    data.load_split()

    valid_evaluators: Dict[str, Evaluator] = {
        dataset_name: Evaluator(
            dataset_df=data.val_by_dataset[dataset_name],
            cutoff_list=cutoff_list,
            batch_size=batch_size,
            verbose=verbose,
        )
        for dataset_name in dataset_names
    }

    test_evaluators: Dict[str, Evaluator] = {
        dataset_name: Evaluator(
            dataset_df=data.test_by_dataset[dataset_name],
            cutoff_list=cutoff_list,
            batch_size=batch_size,
            verbose=verbose,
        )
        for dataset_name in dataset_names
    }

    merged_valid_evaluator = Evaluator(
        dataset_df=data.val_df,
        cutoff_list=cutoff_list,
        batch_size=batch_size,
        verbose=verbose,
    )
    merged_test_evaluator = Evaluator(
        dataset_df=data.test_df,
        cutoff_list=cutoff_list,
        batch_size=batch_size,
        verbose=verbose,
    )

    target_metric = metric_key(metric_name, cutoff)
    prefix = f"merged_{'_'.join(dataset_names)}_{model_name}_{valid_mode}"

    def objective(trial: optuna.Trial):
        params = model_cls.sample_params(trial)
        model = model_cls(**params)
        model.fit(data)

        if valid_mode == "merged":
            merged_metrics = merged_valid_evaluator.evaluate_recommender(model)
            score = merged_metrics[target_metric]
            for k, v in merged_metrics.items():
                trial.set_user_attr(f"merged_valid_{k}", float(v))
            return float(score)

        per_dataset_scores = []
        for dataset_name in dataset_names:
            metrics_dict = valid_evaluators[dataset_name].evaluate_recommender(model)
            per_dataset_scores.append(metrics_dict[target_metric])
            for k, v in metrics_dict.items():
                trial.set_user_attr(f"{dataset_name}_valid_{k}", float(v))

        score = float(np.mean(per_dataset_scores))
        trial.set_user_attr("macro_valid_score", score)
        return score

    study = create_study(prefix=prefix, model_cls=model_cls)
    study.optimize(objective, n_trials=num_trials)

    valid_df = study.trials_dataframe()
    valid_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_valid.csv"), index=False)

    best_params = study.best_params.copy()
    best_model = model_cls(**best_params)
    best_model.fit(data)

    test_rows = []

    merged_test_metrics = merged_test_evaluator.evaluate_recommender(best_model)
    test_rows.append({"dataset": "merged_all", **merged_test_metrics})

    for dataset_name in dataset_names:
        metrics_dict = test_evaluators[dataset_name].evaluate_recommender(best_model)
        test_rows.append({"dataset": dataset_name, **metrics_dict})

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_test.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, f"{prefix}_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print()
    print(f"best value: {study.best_value}")
    print(f"best params: {best_params}")
    print()
    print(test_df.to_string(index=False))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgbm_ranker")
    parser.add_argument("--metric", type=str, default="recall")
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--datasets",
        type=str,
        default="dunnhumby,instacart,tafeng",
        help="comma-separated list",
    )
    parser.add_argument(
        "--valid-mode",
        type=str,
        default="macro",
        choices=["macro", "merged"],
        help="macro = average metric over datasets, merged = metric on union validation",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    run_merged_experiment(
        model_name=args.model,
        metric_name=args.metric,
        cutoff=args.cutoff,
        num_trials=args.num_trials,
        batch_size=args.batch_size,
        dataset_names=dataset_names,
        valid_mode=args.valid_mode,
        verbose=True,
    )