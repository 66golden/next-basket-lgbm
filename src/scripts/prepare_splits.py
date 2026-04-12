from src.dataset import DunnhumbyDataset, InstacartDataset, TafengDataset

for ds in [TafengDataset(), DunnhumbyDataset(), InstacartDataset()]:
    ds.preprocess()
    ds.make_leave_one_basket_split(test_users_rate=0.5, random_basket=False, random_state=42)
    print(f"{ds} split done")
