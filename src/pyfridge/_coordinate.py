"""Coordinate descent for Fridge objective: RSS/(2n) + lambda P_m + tau ||beta||^2/2."""

from dataclasses import dataclass
import warnings

import numpy as np

from ._data import (prepare, validate_m, nonnegative, positive_int, vector,
                    ridge_start, resolve_weights)
from ._penalty import _P_minus_one, penalty, threshold_factors


# Load the optional native kernel once. Import failure selects the Python fallback.
try:
    from ._cd_fast import sweep as _native_sweep
except ImportError:
    # Installation and fitting still work without a compiled extension.
    _native_sweep = None


# Pure python fallback. It uses the same exact updates as the compiled kernel.
def _sweep(X, y, beta, residual, weights, curvature, order,
                  m, lam, tau, zero_tol, block_size):

    # Keep the residual y - X @ beta current throughout a sweep by partial update. 
    # No full recalculation is needed after each coordinate update.
    n, p = X.shape
    max_change = 0.0

    # Process consecutive slices of `order``. Coefficients outside the current
    # block stay fixed, so their polynomial can be cached for the block.
    for start in range(0, p, block_size):
        block = order[start:(start+block_size)]
        outside = np.ones(p, dtype=bool)
        outside[block] = False

        # Store P_{-1},...,P_{m-1} for weighted absolute coefficients outside this block.
        exterior = _P_minus_one(weights[outside] * np.abs(beta[outside]), m) if lam else None

        # Update every coordinate in the block, including coefficients currently zero.
        for j in block:
            factor = 0.0
            if lam:
                others = block[block != j]

                # Recompute within-block terms using the latest updates, excluding coordinate j.
                interior = _P_minus_one(weights[others] * np.abs(beta[others]), m)

                # Combine the cached exterior polynomial with the current within-block
                # polynomial to obtain P_{m-1} for all coordinates except j.
                # Multiplying by lambda and w_j gives the conditional soft-threshold level.
                factor = lam * weights[j] * np.dot(exterior, interior[::-1])
            beta_j_old = beta[j]

            # Add back the current coordinate contribution to obtain its partial-residual
            # score. curvature includes tau, so subtract tau when adding back the data term.
            z = X[:, j] @ residual / n + beta_j_old * (curvature[j] - tau)

            # soft-threshold the score, then divide by the data curvature plus ridge term.
            beta_j_new = (np.sign(z) * max(abs(z) - factor, 0.0) / curvature[j]
                   if curvature[j] > 0 else 0.0)

            # Apply optional truncation before updating the residual.
            if abs(beta_j_new) < zero_tol:
                beta_j_new = 0.0
            beta[j] = beta_j_new
            delta = beta_j_new - beta_j_old

            # partially update residual by -X[:,j] * delta.
            # Track the maximum absolute coefficient change over the sweep.
            if delta:
                residual -= X[:, j] * delta
                max_change = max(max_change, abs(delta))
    return max_change


# Include inactive coordinates.
def kkt_residual(X, y, beta, m, lam, weights, tau=0.0):
    # Gradient of the normalized squared-error loss plus the ridge penalty.
    grad = X.T @ (X @ beta - y) / len(y) + tau * beta
    thresholds = lam * threshold_factors(beta, m, weights) if lam else np.zeros_like(beta)
    active = beta != 0

    # At zero coordinates, stationarity requires |grad_j| <= threshold_j.
    # At nonzero coordinates, require grad_j + threshold_j * sign(beta_j) == 0.
    violation = np.maximum(np.abs(grad) - thresholds, 0.0)
    violation[active] = np.abs(grad[active] + thresholds[active] * np.sign(beta[active]))
    return float(np.max(violation))


def _objective(X, y, beta, m, lam, weights, tau):
    # Evaluate all objective components in the same units as the CD updates.
    residual = y - X @ beta
    pen = lam * penalty(beta, m, weights) if lam else 0.0
    return float(residual @ residual / (2 * len(y)) + pen + tau * (beta @ beta) / 2)


def _solve_cd(X, y, m, lam, weights, beta, tau, tol, max_iter,
              zero_tol, sorted_cd, backend, block_size):

    # Choose a sweep implementation while retaining identical update formulas.
    if backend not in {"auto", "python", "cython"}:
        raise ValueError("backend must be 'auto', 'python', or 'cython'")
    if backend == "cython" and _native_sweep is None:
        raise ImportError("Cython backend is unavailable; rebuild or use backend='python'")
    sweep = _native_sweep if backend != "python" and _native_sweep is not None else _sweep
    backend_used = "cython" if sweep is _native_sweep else "python"

    # Sweeps update coefficients in place, but caller arrays must remain unchanged.
    # Cache the data curvature and initial residual.
    beta = np.array(beta, dtype=np.float64, copy=True)
    curvature = np.mean(X * X, axis=0) + tau
    beta[curvature == 0] = 0.0
    residual = np.ascontiguousarray(y - X @ beta)
    history = [_objective(X, y, beta, m, lam, weights, tau)]

    # Scale the KKT tolerance by the maximum absolute data score at beta = 0.
    target = tol * max(1.0, float(np.max(np.abs(X.T @ y / len(y)))))
    converged = False
    error = np.inf

    # If m==p, the penalty vanishes. Skip all high-degree computation.
    sweep_m = m if m < beta.size else 0
    update_lambda = lam if m < beta.size else 0.0

    # Repeat full sweeps. Small coefficient movement triggers a KKT check rather
    # than serving as sufficient evidence of stationarity on its own.
    for iteration in range(1, max_iter + 1):
        # Every sweep visits zero coordinates, so dropped predictors can reenter.
        order = np.argsort(-np.abs(beta), kind="stable") if sorted_cd else np.arange(beta.size)
        order = np.ascontiguousarray(order, dtype=np.intp)
        change = sweep(X, y, beta, residual, weights, curvature, order,
                       sweep_m, update_lambda, tau, zero_tol, block_size)
        if not np.isfinite(beta).all() or not np.isfinite(residual).all():
            raise FloatingPointError("CD overflow; rescale data/weights or reduce m/lambda")

        # Record the objective after each sweep for diagnostics.
        value = _objective(X, y, beta, m, lam, weights, tau)
        history.append(value)
        if change <= tol * max(1.0, float(np.max(np.abs(beta)))):
            # Refresh the incrementally updated residual before a possible next sweep.
            residual[:] = y - X @ beta
            error = kkt_residual(X, y, beta, m, lam, weights, tau)
            if error <= target:
                converged = True
                break

    # Even at the iteration limit, return the actual final stationarity violation.
    if not converged:
        error = kkt_residual(X, y, beta, m, lam, weights, tau)
    return beta, iteration, converged, error, np.asarray(history), backend_used


@dataclass
class FridgeResult:
    """A fitted Fridge model and its convergence diagnostics.

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients in original input units.
    intercept_ : float
        Unpenalized intercept in original response units.
    scaled_coef_ : np.ndarray
        Coefficients in working units after preprocessing.
    weights_ : np.ndarray
        Nonnegative penalty weights in working units.
    m : int
        Target support size used for fitting.
    lam : float
        Fridge penalty parameter in working units.
    tau : float
        Optional ridge penalty parameter in working units.
    n_iter_ : int
        Number of coordinate-descent sweeps performed.
    converged_ : bool
        Whether the KKT residual satisfies the convergence tolerance. Convergence
        indicates stationarity, not necessarily global optimality.
    kkt_residual_ : float
        Maximum coordinate-wise KKT violation across active and inactive coefficients.
    objective_history_ : np.ndarray
        Working-space objective values during CD. CV-selected points contain a single
        selected objective.
    backend_ : str
        Actual implementation used: python/cython for CD, numpy for unpolished smooth
        paths.
    scaling_ : object
        Preprocessing metadata with predictor/response means and scales.
    """
    coef_: np.ndarray
    intercept_: float
    scaled_coef_: np.ndarray
    weights_: np.ndarray
    m: int
    lam: float
    tau: float
    n_iter_: int
    converged_: bool
    kkt_residual_: float
    objective_history_: np.ndarray
    backend_: str
    scaling_: object

    # Predict in original units.
    def predict(self, X):
        """Predict responses in original response units.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Finite design matrix.

        Returns
        -------
        prediction : ndarray of shape (n_samples,)
            Responses in original units.
        """

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.coef_.size or not np.isfinite(X).all():
            raise ValueError("X must be finite and have the fitted number of columns")
        return X @ self.coef_ + self.intercept_

    # Backward-compatible representation used by the legacy functions.
    def as_dict(self):
        """Export the fit in the legacy dictionary format.

        Returns
        -------
        result : dict
            Keys: coef (intercept first), intercept, scaled_coef, n_iter, penalty.factor, 
              converged, kkt_residual, m, lam, tau, and scaler. 
            scaler contains xm/xsd predictor means/scales and ym/ysd response mean/scale.
        """

        # Export legacy field names; the coefficient vector stores the intercept first.
        s = self.scaling_
        return {"coef": np.r_[self.intercept_, self.coef_], "intercept": self.intercept_,
                "scaled_coef": self.scaled_coef_.copy(), "n_iter": self.n_iter_,
                "penalty.factor": self.weights_.copy(), "converged": self.converged_,
                "kkt_residual": self.kkt_residual_, "m": self.m, "lam": self.lam, "tau": self.tau,
                "scaler": {"xm": s.x_mean, "xsd": s.x_scale,
                           "ym": s.y_mean, "ysd": s.y_scale}}


# This version optimizes RSS/(2n), with unpenalized intercept by default.
def fit_fridge(X, y, m=1, lam=0.1, *, weights=None, adaptive=True, gamma=1.0,
               beta_init=None, fit_intercept=True, standardize=True, scale_y=False,
               tau=0.0, tol=1e-7, max_iter=10000, zero_tol=0.0, sorted_cd=True,
               backend="auto", block_size=None, ridge_init=0.05, weight_epsilon=1e-5,
               warn=True):
    """Fit a Fridge model by coordinate descent.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix.
    y : array_like of shape (n_samples,)
        Finite response vector.
    m : int, default=1
        Target model size, between 0 and n_features. m=0 is weighted Lasso. This is 
        not a hard support constraint.
    lam : float, default=0.1
        Regularization parameter for the Fridge penalty. Objective is 
        RSS/(2n) + lam * P_m + tau * ||beta||^2/2.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative penalty weights in working coefficient units. Explicit weights
        override automatic weight computation; None uses ones unless adaptive=True.
    adaptive : bool, default=True
        Estimate adaptive weights from an initial ridge fit when weights is None.
    gamma : float, default=1.0
        Nonnegative adaptive-weight exponent:
        (abs(beta_ridge)+weight_epsilon)**(-gamma).
    beta_init : array_like of shape (n_features,) or None, default=None
        Initial coefficients in working, centered/scaled units. None uses ridge
        initialization.
    fit_intercept : bool, default=True
        Center X and y and recover an unpenalized intercept. False fixes the intercept
        to zero.
    standardize : bool, default=True
        Scale working predictor columns by their root mean square, after optional
        centering. Returned coefficients use original units.
    scale_y : bool, default=False
        Scale the working response by its root mean square after optional centering;
        outputs are restored to original units.
    tau : float, default=0.0
        Additional ridge regularization parameter, tau*||beta||^2/2, in working units.
    tol : float, default=1e-07
        Positive KKT tolerance, scaled by max(1, ||X.T @ y / n||_inf) in working units.
    max_iter : int, default=10000
        Maximum positive number of CD sweeps during fitting.
    zero_tol : float, default=0.0
        Nonnegative cutoff that sets small coordinate updates to zero. Positive values
        may prevent satisfying the KKT tolerance.
    sorted_cd : bool, default=True
        Visit coordinates in descending absolute coefficient order at each sweep.
    backend : {"auto", "python", "cython"}, default='auto'
        CD implementation. "auto" uses the optional Cython extension when available.
        Otherwise falls back to pure Python.
    block_size : int or None, default=None
        Positive CD block size for caching penalty calculations. None uses max(1,
        int(sqrt(n_features))).
    ridge_init : float, default=0.05
        Nonnegative normalized ridge strength for initialization and adaptive-weight
        pilots. Zero uses least squares.
    weight_epsilon : float, default=1e-05
        Positive offset in adaptive weights, preventing division by zero.
    warn : bool, default=True
        Emit RuntimeWarning for CD fits that fail the KKT convergence check.

    Returns
    -------
    result : FridgeResult
        Fitted coefficients and intercept in original units, working coefficients, weights, iteration count, KKT residual, convergence flag, objective history, backend, and scaling metadata.

    Notes
    -----
    For m > 0, the objective is generally nonconvex, so convergence to a global
    minimum is not guaranteed. Inspect the KKT residual and convergence flag.
    The penalty is applied in working units after preprocessing; returned
    coefficients and predictions are in the original units.
    """
    # 1. Preprocess the data and validate model and solver parameters.
    X, y, scaling = prepare(X, y, fit_intercept, standardize, scale_y)
    p = X.shape[1]
    m = validate_m(m, p)
    lam = nonnegative(lam, "lam")
    
    # Evaluate the single-fit warning after validating p, m, and lam.
    if warn and 0 < m < p and lam > 0:
        warnings.warn(
            "Single-lambda Fridge optimization may converge to a suboptimal "
            "local minimum. Consider using path_fridge() to fit a regularization "
            "path and extract the solution at the desired lambda.",
            UserWarning,
            stacklevel=2,
        )

    tau = nonnegative(tau, "tau")
    tol = nonnegative(tol, "tol", positive=True)
    zero_tol = nonnegative(zero_tol, "zero_tol")
    max_iter = positive_int(max_iter, "max_iter")
    ridge_init = nonnegative(ridge_init, "ridge_init")
    
    # Block size for block-wise CD
    block_size = max(1, int(np.sqrt(p))) if block_size is None else positive_int(block_size, "block_size")

    # 2. Construct weights and initialize the coefficient vector.
    # Predictors that vanish after preprocessing cannot contribute to the fit.
    w = resolve_weights(X, y, weights, adaptive, gamma, ridge_init, weight_epsilon)
    beta = ridge_start(X, y, ridge_init) if beta_init is None else vector(beta_init, p, "beta_init")
    beta[scaling.constant] = 0

    # 3. Run the shared CD engine and retain its actual backend and diagnostics.
    beta, it, ok, error, history, used = _solve_cd(
        X, y, m, lam, w, beta, tau, tol, max_iter, zero_tol, sorted_cd, backend, block_size)
    if warn and not ok:
        warnings.warn(f"Fridge did not converge in {it} sweeps (KKT residual={error:.3g})",
                      RuntimeWarning, stacklevel=2)

    # 4. Undo predictor/response scaling and recover the intercept for user-facing results.
    coef, intercept = scaling.restore(beta)
    return FridgeResult(
        coef_=coef, intercept_=float(intercept), scaled_coef_=beta,
        weights_=w, m=m, lam=lam, tau=tau,
        n_iter_=it, converged_=ok, kkt_residual_=error, objective_history_=history,
        backend_=used, scaling_=scaling
    )
