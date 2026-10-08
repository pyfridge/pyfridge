"""predictor-corrector paths and warm-started CD paths."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import solve

from ._data import (prepare, validate_m, nonnegative, positive_int, vector,
                    ridge_start, resolve_weights)
from ._penalty import penalty_derivatives
from ._coordinate import _solve_cd, kkt_residual, _objective


def _diagonal_solve(X, rhs, diagonal):
    n, p = X.shape

    # Floor the diagonal for numerical stability when the approximation has zero entries.
    # The computed direction uses a regularized approximation of the intended problem.
    diagonal = np.maximum(diagonal, 1e-9)

    # 1. For narrow designs, solve the problem X.T@X/n + D directly.
    if p <= n:
        system = X.T @ X / n

        # Stride p+1 visits only the diagonal entries without creating a dense diagonal matrix.
        system.flat[::p + 1] += diagonal
        return solve(system, rhs, assume_a="pos", check_finite=False)

    # 2. For wide designs use the Woodbury identity, solving an n-by-n system
    # Here, inverse is the diagonal of D^{-1}.
    inverse = 1.0 / diagonal
    weighted_X = X * inverse
    system = weighted_X @ X.T / n
    system.flat[::n + 1] += 1.0
    base = inverse * rhs

    # Recover (D + X.T @ X / n)^{-1} rhs from the sample-space solution.
    return base - inverse * (X.T @ solve(
        system, X @ base / n, assume_a="pos", check_finite=False))


def _restricted_derivatives(beta, m, weights):
    # With at most m variables, every P_k with k >= m penalty is zero.
    if beta.size <= m:
        return np.zeros_like(beta), None
    return penalty_derivatives(beta, m, weights, compute_hessian=False)


def _correct(X, y, beta, m, lam, weights, tau, steps, lr, epsilon, tol):
    # The data, penalty weights, and regularization parameters stay fixed here.
    # An accepted trial already supplies the objective at the next iterate.
    value = _objective(X, y, beta, min(m, beta.size), lam, weights, tau)

    # Recompute the gradient and LQA approximation after each accepted correction.
    # The active design and weights are supplied by the enclosing path algorithm.
    for _ in range(steps):
        # gradient of the penalty
        grad, _ = _restricted_derivatives(beta, m, weights)

        # gradient of the objective
        obj_grad = X.T @ (X @ beta - y) / len(y) + tau * beta + lam * grad
        if np.max(np.abs(obj_grad), initial=0.0) <= tol:
            break

        # Approximate penalty curvature by |gradient_j|/(|beta_j|+epsilon),
        # then add the ridge curvature. Solve for a descent correction.
        diagonal = tau + lam * np.abs(grad) / (np.abs(beta) + epsilon)
        delta = _diagonal_solve(X, -obj_grad, diagonal)

        # Compare trial steps against the cached original penalized objective.
        step = lr

        # Backtrack the damping factor at most 20 times to accept only non-increasing steps, 
        # up to floating-point tolerance.
        # The tiny relative allowance accommodates floating-point objective roundoff.
        for _ in range(20):
            proposed = beta + step * delta
            candidate = _objective(X, y, proposed, min(m, beta.size), lam, weights, tau)
            if np.isfinite(candidate) and candidate <= value + 1e-12 * max(1.0, abs(value)):
                break
            step *= 0.5

        # The loop exhausted all trials without an acceptable step; retain the current fit.
        else:
            break

        # Accept the trial, then stop if its largest coordinate displacement is small.
        beta = proposed
        value = candidate
        if np.max(np.abs(step * delta), initial=0.0) <= tol:
            break
        
    return beta

@dataclass
class FridgePath:
    """A sequence of Fridge candidate fits and path diagnostics.

    Attributes
    ----------
    lambdas_ : np.ndarray
        Evaluated Fridge penalty parameters in working units, shape (n_lambdas,).
    coef_path_ : np.ndarray
        Original-unit coefficient rows, shape (n_lambdas, n_features).
    intercepts_ : np.ndarray
        Original-unit intercepts, shape (n_lambdas,).
    scaled_coef_path_ : np.ndarray
        Working-unit coefficient rows, shape (n_lambdas, n_features).
    weights_ : np.ndarray
        Nonnegative penalty weights in working units.
    m : int
        Target model size used for fitting.
    method : str
        Path algorithm: smooth or cd.
    converged_ : np.ndarray
        KKT tolerance flags, shape (n_lambdas,); do not establish global optimality.
    kkt_residuals_ : np.ndarray
        KKT residual at each path point, shape (n_lambdas,).
    objectives_ : np.ndarray
        Working-space objective at each path point, shape (n_lambdas,).
    n_iter_ : np.ndarray
        CD sweep counts, shape (n_lambdas,); zero for unpolished smooth paths.
    scaling_ : object
        Preprocessing metadata with predictor/response means and scales.
    backend_ : {"python", "cython", "numpy"}, default="numpy"
        Implementation used to compute the path. Unpolished smooth paths
        use NumPy; CD-based paths or polishing use pure Python or Cython.
    """
    # Coefficients are rows (lambda points) by columns (predictors), without ambiguity.
    lambdas_: np.ndarray
    coef_path_: np.ndarray
    intercepts_: np.ndarray
    scaled_coef_path_: np.ndarray
    weights_: np.ndarray
    m: int
    method: str
    converged_: np.ndarray
    kkt_residuals_: np.ndarray
    objectives_: np.ndarray
    n_iter_: np.ndarray
    scaling_: object
    backend_: str = "numpy"

    @property
    def model_sizes_(self):
        """Return the number of nonzero coefficients at each path point.

        Returns
        -------
        sizes : ndarray of shape (n_lambdas,)
            Nonzero coefficient counts for each path point.
        """
        return np.count_nonzero(self.coef_path_, axis=1)

    # Prediction can select one lambda or return an observations-by-lambdas matrix.
    def predict(self, X, index=None):
        """Predict responses in original response units.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Finite design matrix.
        index : int, slice, 1-D array_like of int or bool, or None, default=None
            Path row or rows to predict from. Integer sequences preserve their
            order and duplicates; boolean masks select matching path rows.
            None returns predictions for every lambda value.

        Returns
        -------
        prediction : ndarray of shape (n_samples,) or (n_samples, n_selected)
            Responses in original units. An integer index returns a vector;
            multiple indices, slices, and masks return one column per selected
            point, including shape (n_samples, 1) for a one-element list.
            With index=None, n_selected is the full number of path points.
        """

        # Check prediction dimensions against the fitted path before matrix multiplication.
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.coef_path_.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be finite and have the fitted number of columns")

        # Broadcast intercepts over observations; each output column corresponds to a lambda.
        if index is None:
            return X @ self.coef_path_.T + self.intercepts_
        
        # A scalar selects a length-p vector, whose transpose leaves it unchanged.
        # A sequence selects an (n_selected, p) matrix; transposition is needed
        # for X @ coefficients to yield an (n_samples, n_selected) prediction.
        if not isinstance(index, slice):
            index = np.asarray(index)
            if index.ndim > 1 or (index.ndim == 0 and index.dtype.kind == "b"):
                raise ValueError("index must be an integer, slice, or one-dimensional index array")
            if index.size == 0:
                index = np.empty(0, dtype=np.intp)

        return X @ self.coef_path_[index].T + self.intercepts_[index]


def _lambda_grid(X, y, m, w, lambdas, n_steps, lambda_init, lambda_max, method):
    # An explicit grid takes precedence over all automatic-grid controls.
    # Smooth prediction advances forward in lambda, so its grid must increase strictly.
    if lambdas is not None:
        grid = np.asarray(lambdas, dtype=float)
        if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
            raise ValueError("lambdas must be a nonempty, finite, nonnegative vector")
        if method == "smooth" and np.any(np.diff(grid) <= 0):
            raise ValueError("smooth requires strictly increasing lambdas")
        return grid.copy()

    # Generate a geometric grid in normalized objective units when none was supplied.
    n_steps = positive_int(n_steps, "n_steps")
    start = 1e-6 / (len(y) * max(m, 1)) if lambda_init is None else nonnegative(
        lambda_init, "lambda_init", positive=True)

    # Use data scores for the automatic endpoint. Only m=0 admits the usual
    # weighted-Lasso zero-fit threshold; for higher degrees this is a heuristic.
    if lambda_max is None:
        scores = np.abs(X.T @ y / len(y))
        end = float(np.max(scores / np.maximum(w, 1e-12))) if m == 0 else float(np.max(scores))
    else:
        end = nonnegative(lambda_max, "lambda_max")

    # Handle degenerate or non-increasing automatic ranges.
    if end == 0:
        return np.array([0.0])
    if end <= start:
        return np.array([start])
    return np.geomspace(start, end, n_steps + 1)


def path_fridge(X, y, m=1, lambdas=None, *, method="smooth", weights=None,
                adaptive=True, gamma=1.0, n_steps=100, lambda_init=None,
                lambda_max=None, correction_steps=2,
                initial_correction_steps=10, polish=False, stop_at_target=True,
                threshold=1e-3, beta_init=None, fit_intercept=True,
                standardize=True, scale_y=False, tau=0.0, tol=1e-7,
                max_iter=10000, backend="auto", block_size=None,
                ridge_init=1e-6, weight_epsilon=1e-5, lqa_epsilon=1e-6, warn=True):
    """Compute a smooth predictor-corrector or coordinate-descent solution path.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix.
    y : array_like of shape (n_samples,)
        Finite response vector.
    m : int, default=1
        Target model size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    lambdas : array_like of shape (n_lambdas,) or None, default=None
        Finite nonnegative normalized penalty multipliers. Smooth requires strictly
        increasing values; CD accepts any order. Explicit grids are evaluated in full.
        None generates a geometric grid.
    method : {"smooth", "cd"}, default='smooth'
        Smooth uses the approximate predictor-corrector with a diagonal quadratic
        Hessian approximation; cd uses warm-started coordinate descent.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative penalty weights in working coefficient units. Explicit weights
        override adaptive weighting; None uses ones unless adaptive=True.
    adaptive : bool, default=True
        Estimate weights from a ridge pilot when weights is None.
    gamma : float, default=1.0
        Nonnegative adaptive-weight exponent:
        (abs(beta_ridge)+weight_epsilon)**(-gamma).
    n_steps : int, default=100
        Positive number of intervals in an automatic grid, normally giving n_steps+1
        points. Degenerate grids may contain one point; smooth stopping can shorten
        paths. Ignored for explicit lambdas.
    lambda_init : float or None, default=None
        Positive starting multiplier for automatic grids. None uses
        1e-6/(n_samples*max(m,1)); ignored for explicit lambdas.
    lambda_max : float or None, default=None
        Nonnegative endpoint for automatic grids. None uses a score-based heuristic; for
        m>0 this is not a guaranteed all-zero-fit bound. Ignored for explicit lambdas.
    correction_steps : int, default=2
        Maximum positive number of repeated LQA corrections after each smooth predictor
        step; early stopping may use fewer.
    initial_correction_steps : int, default=10
        Maximum positive number of LQA corrections at the initial smooth path point.
    polish : bool, default=False
        Run full CD after each smooth candidate, allowing deleted coordinates to
        reenter.
    stop_at_target : bool, default=True
        Stop an automatically generated smooth path when at most m coefficients remain.
        Explicit grids always run in full.
    threshold : float, default=0.001
        Nonnegative working-unit cutoff for deleting small coefficients in smooth paths.
        Deleted coordinates cannot reenter unless polish=True.
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
        Nonnegative additional ridge multiplier, tau*||beta||^2/2, in working units.
    tol : float, default=1e-07
        Positive KKT tolerance, scaled by max(1, max(abs(X.T@y/n))) in working units.
        Smooth corrections also use this as a stopping tolerance.
    max_iter : int, default=10000
        Maximum positive number of CD sweeps per fit or path point; used by method="cd"
        or polishing.
    backend : {"auto", "python", "cython"}, default='auto'
        CD implementation. Auto uses the optional compiled extension when available;
        cython requires it. Unpolished smooth paths use NumPy/SciPy.
    block_size : int or None, default=None
        Positive CD block size for caching penalty calculations. None uses max(1,
        int(sqrt(n_features))).
    ridge_init : float, default=1e-06
        Nonnegative normalized ridge strength for initialization and adaptive-weight
        pilots. Zero uses least squares.
    weight_epsilon : float, default=1e-05
        Positive offset in adaptive weights, preventing division by zero.
    lqa_epsilon : float, default=1e-06
        Positive numerical offset in the approximation diag(abs(grad
        P_m)/(abs(beta)+lqa_epsilon)).
    warn : bool, default=True
        Emit RuntimeWarning for CD fits that fail the KKT convergence check.

    Returns
    -------
    path : FridgePath
        Lambda grid; original-unit and working coefficient arrays of shape (n_lambdas, n_features); intercepts, KKT residuals, convergence flags, objectives, and CD sweep counts of shape (n_lambdas,); weights and scaling metadata. Smooth-only sweep counts are zero.

    Notes
    -----
    Smooth predictors and correctors allow coefficients to change sign.
    Magnitude thresholding removes small coefficients between path points.
    Corrector candidates are checked against the original objective.
    Deleted coordinates can reenter through polishing.
    For m>0, global optimality is not guaranteed. Inspect KKT residuals and
    convergence flags; smooth paths are approximate candidates. Penalties apply
    in working units after preprocessing; returned predictions use original units.
    """

    # Validate the algorithm choice, preprocessing, and numerical controls once per path.
    if method not in {"smooth", "cd"}:
        raise ValueError("method must be 'smooth' or 'cd'")
    X, y, scaling = prepare(X, y, fit_intercept, standardize, scale_y)
    n, p = X.shape
    m = validate_m(m, p)
    tau = nonnegative(tau, "tau")
    tol = nonnegative(tol, "tol", positive=True)
    threshold = nonnegative(threshold, "threshold")
    ridge_init = nonnegative(ridge_init, "ridge_init")
    lqa_epsilon = nonnegative(lqa_epsilon, "lqa_epsilon", positive=True)
    max_iter = positive_int(max_iter, "max_iter")
    correction_steps = positive_int(correction_steps, "correction_steps")
    initial_correction_steps = positive_int(initial_correction_steps, "initial_correction_steps")
    block_size = max(1, int(np.sqrt(p))) if block_size is None else positive_int(block_size, "block_size")

    # Keep penalty weights fixed along the path; initialize from ridge or the user warm start.
    w = resolve_weights(X, y, weights, adaptive, gamma, ridge_init, weight_epsilon)
    grid = _lambda_grid(X, y, m, w, lambdas, n_steps, lambda_init, lambda_max, method)
    beta = ridge_start(X, y, ridge_init) if beta_init is None else vector(beta_init, p, "beta_init")
    beta[scaling.constant] = 0

    # Accumulate copies of coefficients and diagnostics so later updates cannot alter earlier points.
    paths, errors, objectives, iterations, checks = [], [], [], [], []
    target = tol * max(1.0, float(np.max(np.abs(X.T @ y / n))))

    # Use the rule-of-thumb damping schedule, capped at one and floored at 0.1.
    lr = max(min(1.0, 4.0 / (m + 1.0)), 0.1)
    
    # Smooth uses NumPy/SciPy; CD/polishing records the selected sweep backend.
    backend_used = "numpy"

    # Advance through the lambda grid, warm-starting each point from the previous coefficients.
    for index, lam in enumerate(grid):
        steps = 0
        if method == "smooth":

            # Select coefficients remaining above the working-unit threshold. Without
            # polishing, smooth continuation operates only on this surviving support.
            active = np.flatnonzero(np.abs(beta) >= threshold) if threshold else np.flatnonzero(beta)
            beta[np.abs(beta) < threshold] = 0
            if active.size:

                # Restrict the design, coefficients, and weights to matching active coordinates.
                XA, bA, wA = X[:, active], beta[active].copy(), w[active]

                # The first point receives corrections only. Later points first take an Euler
                # predictor step using the previous lambda and its current LQA curvature.
                if index:
                    # quadratic approximation in predictor step.
                    grad, _ = _restricted_derivatives(bA, m, wA)
                    diagonal = tau + grid[index - 1] * np.abs(grad) / (np.abs(bA) + lqa_epsilon)
                    velocity = _diagonal_solve(XA, -grad, diagonal)

                    # Multiply d beta/d lambda by the lambda increment to predict the next fit.
                    proposed = bA + (lam - grid[index - 1]) * velocity
                    # Retain sign changes; prune only zeros and small magnitudes.
                    keep = (proposed != 0) & (np.abs(proposed) >= threshold)

                    # Clear the old active entries before writing back only the surviving coordinates.
                    beta[active] = 0
                    active, proposed, wA = active[keep], proposed[keep], wA[keep]
                    XA = X[:, active]
                    bA = proposed

                # Correct the predictor at the new lambda; give the initial point a separate correction budget.
                if active.size:
                    bA = _correct(XA, y, bA, m, lam, wA, tau,
                                  initial_correction_steps if index == 0 else correction_steps,
                                  lr, lqa_epsilon, tol)
                    beta[active] = bA
                beta[np.abs(beta) < threshold] = 0

        # Full CD checks all coordinates and can reintroduce variables deleted by the smooth step.
        if method == "cd" or polish:
            # Preserve backend information when returning a fitted path point from CV.
            beta, steps, ok, error, _, backend_used = _solve_cd(
                X, y, m, float(lam), w, beta, tau, tol, max_iter,
                0.0, True, backend, block_size)
        else:

            # Without polishing, measure stationarity directly; the smooth approximation
            # need not meet the requested KKT tolerance even when corrections have stopped.
            error = kkt_residual(X, y, beta, m, float(lam), w, tau)
            ok = error <= target
        if not np.isfinite(beta).all():
            raise FloatingPointError("path overflow; rescale data/weights or reduce m")

        # Save one immutable snapshot and its diagnostic values for this lambda.
        paths.append(beta.copy())
        errors.append(error)
        checks.append(ok)
        objectives.append(_objective(X, y, beta, m, float(lam), w, tau))
        iterations.append(steps)

        # Early support-based termination applies only to automatic smooth grids.
        # User-specified grids must preserve every requested lambda point.
        if (method == "smooth" and stop_at_target and lambdas is None
                and np.count_nonzero(beta) <= m):
            break
    if warn and method == "cd" and not all(checks):
        warnings.warn("Some CD path points did not converge; inspect converged_ and kkt_residuals_",
                      RuntimeWarning, stacklevel=2)

    # Restore all coefficient rows together and truncate the grid to the points actually evaluated.
    scaled = np.asarray(paths)
    coef, intercept = scaling.restore(scaled)
    return FridgePath(grid[:len(paths)], coef, intercept, scaled, w, m, method,
                      np.asarray(checks), np.asarray(errors), np.asarray(objectives),
                      np.asarray(iterations), scaling, backend_=backend_used)


# Explicit separate operation for final OLS refitting.
def refit_ols(X, y, coef, *, fit_intercept=True):
    """Refit ordinary least squares on a supplied nonzero support.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix.
    y : array_like of shape (n_samples,)
        Finite response vector.
    coef : array_like of shape (n_features,)
        Finite coefficient vector whose nonzero entries specify the support for OLS
        refitting. Its magnitudes do not affect the refit.
    fit_intercept : bool, default=True
        Center X and y and recover an unpenalized intercept. False fixes the intercept
        to zero.

    Returns
    -------
    coef : ndarray of shape (n_features,)
        OLS coefficients in original units, zero outside the supplied support.
        Rank-deficient supports use a minimum-norm least-squares solution.
    intercept : float
        Fitted intercept, or zero when fit_intercept=False.
    """

    # Center if requested, but do not rescale: OLS refitting uses the selected support only.
    X, y, s = prepare(X, y, fit_intercept, False, False)
    coef = vector(coef, X.shape[1], "coef")

    # Coefficient magnitudes only choose support. LS estimates new values on it.
    support = coef != 0
    beta = np.zeros_like(coef)

    # Use a least-squares solve, including rank-deficient supports.
    if support.any():
        beta[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
    return s.restore(beta)
