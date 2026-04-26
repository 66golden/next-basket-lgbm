from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .dunnhumby import DunnhumbyDataset
from .instacart import InstacartDataset
from .tafeng import TafengDataset


SOURCE_DATASETS = {
    "dunnhumby": DunnhumbyDataset,
    "instacart": InstacartDataset,
    "tafeng": TafengDataset,
}


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class MergedAllDataset:
    def __init__(
        self,
        dataset_names: List[str] | None = None,
        verbose: bool = False,
    ):
        self.dataset_names = dataset_names or ["dunnhumby", "instacart", "tafeng"]
        self.verbose = verbose

        self.dataset_dir = "merged_all__" + "__".join(self.dataset_names)

        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.val_by_dataset: Dict[str, pd.DataFrame] = {}
        self.test_by_dataset: Dict[str, pd.DataFrame] = {}

        self._index2user: Dict[int, str] | None = None
        self._user2index: Dict[str, int] | None = None
        self._index2item: Dict[int, str] | None = None
        self._item2index: Dict[str, int] | None = None

    @property
    def num_users(self):
        return 0 if self._user2index is None else len(self._user2index)

    @property
    def num_items(self):
        return 0 if self._item2index is None else len(self._item2index)

    def _print(self, msg: str):
        if self.verbose:
            print(f"{type(self).__name__}: {msg}")

    def _load_source_dataset(self, name: str):
        if name not in SOURCE_DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        ds = SOURCE_DATASETS[name](verbose=self.verbose)
        ds.load_split()
        return ds

    def _denorm_split(self, dataset_name: str, ds, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["user_key"] = out["user_id"].map(ds._index2user).map(
            lambda x: f"{dataset_name}::u::{x}"
        )

        def map_basket(basket):
            return [
                f"{dataset_name}::i::{ds._index2item[int(item_id)]}"
                for item_id in basket
            ]

        out["basket_key"] = out["basket"].apply(map_basket)
        out["timestamp"] = pd.to_datetime(out["timestamp"])

        return out.loc[:, ["user_key", "basket_key", "timestamp"]]

    def _build_global_mappings(
        self,
        all_splits: List[pd.DataFrame],
    ):
        all_user_keys = set()
        all_item_keys = set()

        for df in all_splits:
            all_user_keys.update(df["user_key"].unique().tolist())
            for basket in df["basket_key"]:
                all_item_keys.update(basket)

        user_keys = sorted(all_user_keys)
        item_keys = sorted(all_item_keys)

        self._user2index = {user_key: idx for idx, user_key in enumerate(user_keys)}
        self._index2user = {idx: user_key for user_key, idx in self._user2index.items()}

        self._item2index = {item_key: idx for idx, item_key in enumerate(item_keys)}
        self._index2item = {idx: item_key for item_key, idx in self._item2index.items()}

    def _remap_split(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "user_id": df["user_key"].map(self._user2index).astype(int),
                "basket": df["basket_key"].apply(
                    lambda basket: [int(self._item2index[item_key]) for item_key in basket]
                ),
                "timestamp": pd.to_datetime(df["timestamp"]),
            }
        )
        out = out.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
        return out

    def load_split(self):
        self._print("Loading source datasets...")

        denorm_train_parts = []
        denorm_val_by_dataset = {}
        denorm_test_by_dataset = {}

        for dataset_name in self.dataset_names:
            ds = self._load_source_dataset(dataset_name)

            denorm_train = self._denorm_split(dataset_name, ds, ds.train_df)
            denorm_val = self._denorm_split(dataset_name, ds, ds.val_df)
            denorm_test = self._denorm_split(dataset_name, ds, ds.test_df)

            denorm_train_parts.append(denorm_train)
            denorm_val_by_dataset[dataset_name] = denorm_val
            denorm_test_by_dataset[dataset_name] = denorm_test

        self._print("Building global mappings...")
        self._build_global_mappings(
            denorm_train_parts
            + list(denorm_val_by_dataset.values())
            + list(denorm_test_by_dataset.values())
        )

        self._print("Remapping splits to global id space...")
        self.train_df = pd.concat(
            [self._remap_split(df) for df in denorm_train_parts],
            axis=0,
            ignore_index=True,
        ).sort_values(["timestamp", "user_id"]).reset_index(drop=True)

        self.val_by_dataset = {
            dataset_name: self._remap_split(df)
            for dataset_name, df in denorm_val_by_dataset.items()
        }
        self.test_by_dataset = {
            dataset_name: self._remap_split(df)
            for dataset_name, df in denorm_test_by_dataset.items()
        }

        self.val_df = pd.concat(
            list(self.val_by_dataset.values()),
            axis=0,
            ignore_index=True,
        ).sort_values(["timestamp", "user_id"]).reset_index(drop=True)

        self.test_df = pd.concat(
            list(self.test_by_dataset.values()),
            axis=0,
            ignore_index=True,
        ).sort_values(["timestamp", "user_id"]).reset_index(drop=True)

        self._print(
            f"Done. num_users={self.num_users}, num_items={self.num_items}, "
            f"train_rows={len(self.train_df)}, val_rows={len(self.val_df)}, test_rows={len(self.test_df)}"
        )
        return self.train_df, self.val_df, self.test_df