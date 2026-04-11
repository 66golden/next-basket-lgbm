# Next-Basket-LGBM

## Setting virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model run:

```bash
PYTHONPATH=. python src/scripts/prepare_splits.py

PYTHONPATH=. python src/scripts/experiment.py --dataset dunnhumby --model lgbm_ranker --num-trials 100
PYTHONPATH=. python src/scripts/experiment.py --dataset tafeng --model lgbm_ranker --num-trials 100
PYTHONPATH=. python src/scripts/experiment.py --dataset instacart --model lgbm_ranker --num-trials 100
```


## License

This project is licensed under the Apache License 2.0. see the `LICENSE` file for details.

## Acknowledgments

This project was developed based on ideas and parts of code structure from the repository `time_dependent_nbr`, which was used as an experimental starting point and then substantially modified.