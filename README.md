# THERA-Net

Thermodynamically admissible neural modeling of rate-and-state friction using NumPy and SciPy.

- `model.py`: neural network, friction equations, and state evolution.
- `material.py`: synthetic material family and prediction helpers.
- `train.py`: training and validation.

## Run

Requires Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
python train.py
```

Training saves the model, datasets, and metrics in `results/`.
For a shorter run, use `python train.py --maxiter 100 --seeds 10`.
