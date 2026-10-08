# pyfridge

**Fridge (Fractional Ridge) regression for Python:** fit sparse regression models, follow regularization paths, and select tuning parameters by cross-validation.

Fridge's target model size `m` specifies the model size toward which regularization is directed; `lam` controls the penalty strength. The number of selected variables can exceed `m`. See [Fractional Ridge Regression](https://arxiv.org/abs/2505.23925) for the method and theory.

## Installation

Python **3.10 or newer** is required. Download or clone this repository, open a terminal in the directory containing `pyproject.toml`, and run:

```sh
python -m pip install .
```

To include plotting support:

```sh
python -m pip install ".[plot]"
```

For the riboflavin notebook in example, including plotting and Jupyter:

```sh
python -m pip install ".[example]"
python -m jupyter lab
```

The package can build an optional Cython coordinate-descent extension when a C compiler is available. A Python fallback is available. To install without compiling the extension on macOS/Linux:

```sh
PYFRIDGE_NO_EXT=1 python -m pip install ".[plot]"
```

On Windows PowerShell, set `$env:PYFRIDGE_NO_EXT = "1"` before installing.

## Get started

Inputs are a design matrix `X` with shape `(n_samples, n_features)` and a response vector `y` with shape `(n_samples,)`. Missing values must be handled before fitting.

```python
import numpy as np
from pyfridge import cv_fridge

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 8))
y = 2 + 1.5 * X[:, 0] - X[:, 2] + rng.normal(scale=0.4, size=100)

cv = cv_fridge(
    X, y, ms=[1, 2, 3], n_folds=5, n_steps=40,
    random_state=42, selection="1se", scale_y=True,
)
print("Selected m and lam:", cv.one_se_m_, cv.one_se_lambda_)
print("Coefficients:", cv.fit_.coef_)
predictions = cv.predict(X)
```

Use held-out data to assess prediction performance. The example predicts the training data only to demonstrate the API.

## Main interface

| Interface | Purpose | Returned object |
|---|---|---|
| `fit_fridge(X, y, m, lam, ...)` | Coordinate-descent fit at a specified penalty | `FridgeResult` |
| `path_fridge(X, y, m, ...)` | Fits across a regularization grid | `FridgePath` |
| `cv_fridge(X, y, ms, ...)` | Selects `m` and `lam` by cross-validation | `FridgeCV` |
| `Fridge(m=..., lam=..., ...)` | Estimator-style interface with `fit`, `predict`, and `score` | Fitted `Fridge` instance |

### Fit at a specified penalty

`fit_fridge()` uses coordinate descent. Because the Fridge objective is generally nonconvex for m > 0, single-lambda optimization may converge to a suboptimal local minimum. **We recommend fitting a regularization path with `path_fridge()` and extracting the solution at the desired lambda.** `fit_fridge()` primarily serves as an interface to `path_fridge()` with `method="cd"`.

### Tracing a solution path

```python
from pyfridge import path_fridge

grid = np.geomspace(1e-7, 0.1, 60)
path = path_fridge(X, y, m=2, lambdas=grid, scale_y=True)
print(path.coef_path_.shape)  # (n_lambdas, n_features)
print(path.model_sizes_)
y_all = path.predict(X)              # (n_samples, n_lambdas)
y_some = path.predict(X, index=[0, 2])
```

The default `method="smooth"` uses a predictor–corrector with a quadratic Hessian approximation. Set `method="cd"` for warm-started coordinate descent. `lam` grids for `method="smooth"` must strictly increase. With an automatically generated grid, the path can stop once at most `m` variables remain; explicit grids are evaluated in full.

### Choosing regularization parameters

`cv_fridge` defaults to the `"smooth"` method. The `"min"` rule selects the lowest mean fold MSE. The `"1se"` rule selects the smallest admissible `m`, then the largest `lam` within the global one-standard-error threshold.

```python
print("Minimum:", cv.best_m_, cv.best_lambda_)
print("One-SE:", cv.one_se_m_, cv.one_se_lambda_)
print("Selected model diagnostics:", cv.fit_.converged_, cv.fit_.kkt_residual_)

# The final model already uses the selected fit from its full-data path.
selected_path = cv.path_
selected_lam = cv.fit_.lam
index = np.flatnonzero(selected_path.lambdas_ == selected_lam)[0]
selected_coefficients = selected_path.coef_path_[index]
y_pred = selected_path.predict(X, index=index)
```

Scaling and adaptive weights are learned within each training fold. `cv.results_` contains the grids, fold errors, means, standard errors, and convergence diagnostics for every candidate `m`.

### Estimator-style interface

```python
from pyfridge import Fridge

model = Fridge(m=2, lam=0.01, scale_y=True)
model.fit(X, y)
y_pred = model.predict(X)
print(model.coef_, model.intercept_)
print(model.score(X, y))  # R-squared
```

`Fridge` wraps the coordinate-descent fit and exposes `get_params()` and `set_params()`. It does not perform CV automatically. Fitted preprocessing is available through `scaling_`, and the underlying result through `fit_result_`.

## Parameters and diagnostics

The normalized objective is

```text
||y - X beta||² / (2n) + lam * P_m(beta; weights) + tau * ||beta||² / 2.
```

`P_m` sums all products of `m+1` distinct weighted absolute coefficients. `m=0` gives weighted Lasso; `m=n_features` makes the Fridge penalty zero. `tau` is an optional additional L2 penalty, separate from `lam`.

Predictors are centered and standardized by default, with an unpenalized intercept. The response is scaled only when `scale_y=True`. Coefficients and predictions are returned in original units. Penalty parameters act in the working units, so changing preprocessing changes their interpretation.

Adaptive weights are enabled by default. Use `adaptive=False` for unweighted fitting or supply nonnegative `weights` explicitly. `ridge_init` controls the ridge pilot and initialization: its default is `0.05` for single CD fits and `1e-6` for paths/CV. Set it explicitly when comparing procedures.

For `m>0`, the objective is generally nonconvex. Smooth paths are approximate, and coordinate-descent convergence does not generally establish global optimality. Check `converged_` and `kkt_residual_` on a fitted result, or `converged_` and `kkt_residuals_` on a path. `polish=True` applies CD after smooth candidates and can change their support. Use `help(...)` for the full parameter and return-value documentation.

## Plotting

```python
import matplotlib.pyplot as plt
from pyfridge import plot_path_fridge, plot_cv_fridge

ax = plot_path_fridge(cv.path_)  # Working-unit coefficients by default
ax.figure.tight_layout()
ax.figure.savefig("path.pdf", bbox_inches="tight")

ax = plot_cv_fridge(cv)
ax.figure.tight_layout()
ax.figure.savefig("cv.pdf", bbox_inches="tight")
plt.show()
```

`plot_path_fridge` accepts `feature_names`, a subset of column indices through `features`, and `scaled=False` for original-unit coefficients. By default it displays predictors that are nonzero somewhere along the path and suppresses large legends; use `legend=True` to show all labels.

`plot_cv_fridge` displays every candidate `m`, marks the minimum and one-SE selections, and shows the global threshold. `show_se=False` hides the standard-error bands; these bands are not confidence intervals.

Both functions accept an existing Matplotlib `ax` and return it. They never refit models or call `show`. Positive lambda grids use a log-scale axis by default; grids containing zero can use an appropriate symmetric log axis. Set `xscale="linear"` when preferred.

## Reproducing riboflavin example

The example is in [example/riboflavin.ipynb](example/riboflavin.ipynb). It downloads the dataset directly from the public CRAN `hdi` mirror, then runs Fridge, adaptive Lasso, SCAD, and SuSiE, followed by gene-level and correlation-group comparisons. This includes ten-fold CV and the paper-tuned Fridge path, with an optional 500-resample fixed-tuning bootstrap enabled.

The notebook needs internet access (solely to download the riboflavin data), the `example` Python dependencies, and **R with `ncvreg`**. In R, run `install.packages("ncvreg")`, then ensure `Rscript` is on your PATH. Downloads and R exchange files use temporary directories.

The paper's reported `lam` value is converted to the package's normalized objective. Fresh package CV is reported separately because fold construction and one-SE implementation can change the selected parameters. Seeds and software versions are printed. This distinction avoids treating a fresh analysis as an exact reproduction of historical cached results.

## Dependencies

- **Core:** Python 3.10+, NumPy, SciPy.
- **Optional plotting:** Matplotlib (`.[plot]`).
- **Riboflavin example:** Matplotlib, pandas, scikit-learn, pysusie, and Jupyter (`.[example]`); R and `ncvreg` are installed separately.
- **Build:** setuptools, wheel, Cython, and NumPy. A C compiler is optional when using the Python fallback.

## Reference

Park, S. and Stefanski, L. A. **Fractional Ridge Regression**. arXiv:2505.23925v3. [[paper]](https://arxiv.org/pdf/2505.23925).

Please cite the paper when using Fridge in research. Dataset attribution and comparison-method references are included in the riboflavin notebook.
