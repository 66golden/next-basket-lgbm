from src.dataset import DunnhumbyDataset, InstacartDataset, TafengDataset


def main():
    datasets = [
        TafengDataset("tafeng", verbose=True),
        DunnhumbyDataset("dunnhumby", verbose=True),
        InstacartDataset("instacart", verbose=True),
    ]

    for ds in datasets:
        print(f"start {ds}")
        ds.preprocess()
        ds.make_leave_one_basket_split(
            test_users_rate=0.5,
            random_basket=False,
            random_state=42,
        )
        print(f"{ds} split done")


if __name__ == "__main__":
    main()