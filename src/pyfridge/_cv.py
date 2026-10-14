"""Cross-validation at shared normalized lambdas."""

from dataclasses import dataclass
import numpy as np

from ._data import data_arrays, validate_m, positive_int
from ._path import path_fridge
from ._coordinate import FridgeResult


@dataclass
class FridgeCV:
    """Cross-validation choices, fold diagnostics, and the selected full-data fit.

    Attributes
    ----------
    results_ : dict
        Mapping from candidate m to lambdas, fold_mse, mean_mse, se_mse, converged, and
        kkt_residuals arrays. Fold arrays have shape (n_folds, n_lambdas); summaries
        have shape (n_lambdas,).
    best_m_ : int
        Target model size selected by minimum cross-validation error.
    best_lambda_ : float
        Penalty parameter selected by minimum cross-validation error.
    one_se_m_ : int
        Smallest target model size selected by the global one-SE threshold.
    one_se_lambda_ : float
        Largest penalty parameter within the one-SE threshold at `one_se_m_`.
    one_se_threshold_ : float
        The minimum mean fold MSE plus its standard error.
    selection : str: {"min", "1se"}
        Cross-validation selection rule used for the final fit.
    fit_ : FridgeResult
        Selected full-data fit.
    method : str = "smooth". {"smooth", "cd"}
        Algorithm used to fit the cross-validation paths.
    path_ : object
        Full-data FridgePath containing the selected point.
    """
    results_: dict
    best_m_: int
    best_lambda_: float
    one_se_m_: int
    one_se_lambda_: float
    one_se_threshold_: float
    selection: str
    fit_: FridgeResult
    method: str = "smooth"
    path_: object = None

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

        return self.fit_.predict(X)


def cv_fridge(X, y, ms=(0, 1, 2), lambdas=None, *, n_steps=100,
                   n_folds=5, random_state=42, selection="1se", method="smooth",
                   **fit_options):
    """Select Fridge target size and regularization using shuffled cross-validation.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix. Inputs are copied; columns are predictors.
    y : array_like of shape (n_samples,)
        Finite response vector.
    ms : iterable of int, default=(0, 1, 2)
        Nonempty candidate target model sizes, each between 0 and n_features. Duplicates are
        removed and candidates are sorted.
    lambdas : array_like of shape (n_lambdas,) or None, default=None
        Finite nonnegative Fridge penalty parameters in working units. For method="smooth", 
        values must be strictly increasing; method="cd" accepts any order. Explicit grids 
        are evaluated in full. None generates a geometric grid.
    n_steps : int, default=100
        Positive number of intervals in an automatic grid, normally giving n_steps+1
        points. Degenerate grids may contain one point; smooth stopping can shorten
        paths. Automatic full-data refits use 100 intervals, inserting the
        selected lambda when needed. Ignored for explicit lambdas.
    n_folds : int, default=5
        Number of shuffled folds, between 2 and n_samples.
    random_state : int, numpy.random.Generator or None, default=42
        Seed or generator accepted by numpy.random.default_rng for shuffled folds.
    selection : {"min", "1se"}, default="1se"
        Selection rule. "min" chooses the candidate with the smallest mean fold MSE.
        "1se" chooses the smallest m with a candidate within the global 1-SE
        threshold, then the largest lambda within the threshold at that m.
    method : {"smooth", "cd"}, default='smooth'
        "smooth" uses the approximate predictor-corrector with a diagonal quadratic
        Hessian approximation; "cd" uses warm-started coordinate descent.
    **fit_options : keyword arguments
        Additional options passed to `path_fridge`. See Other Parameters.
        `beta_init`, `stop_at_target`, and `exact` are not accepted.

    Other Parameters
    ----------------
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative penalty weights in working coefficient units. Explicit weights
        override adaptive weighting; None uses ones unless adaptive=True.
    adaptive : bool, default=True
        Estimate weights from a ridge pilot when weights is None.
    gamma : float, default=1.0
        Nonnegative adaptive-weight exponent:
        (abs(beta_ridge)+weight_epsilon)**(-gamma).
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
        Nonnegative ridge penalty parameter in tau*||beta||^2/2, in working units.
    tol : float, default=1e-07
        Positive KKT tolerance, scaled by max(1, max(abs(X.T@y/n))) in working units.
        Smooth corrections also use this as a stopping tolerance.
    max_iter : int, default=10000
        Maximum positive number of CD sweeps per fit or path point; used by method="cd"
        or polishing.
    backend : {"auto", "python", "cython"}, default="auto"
        Coordinate-descent backend. "auto" uses the optional Cython extension when
        available and otherwise falls back to Python. Used by method="cd".
    block_size : int or None, default=None
        Positive CD block size for caching penalty calculations. None uses max(1,
        int(sqrt(n_features))).
    ridge_init : float, default=1e-06
        Nonnegative normalized ridge strength for initialization and adaptive-weight
        pilots. Zero uses least squares.
    weight_epsilon : float, default=1e-05
        Positive offset in adaptive weights, preventing division by zero.
    warn : bool, default=True
        Emit RuntimeWarning for CD fits that fail the KKT convergence check.
    correction_steps : int, default=2
        Maximum number of repeated LQA corrections after each smooth predictor step. 
        Corrections may stop early.
    initial_correction_steps : int, default=10
        Maximum positive number of LQA corrections at the initial smooth path point.
    polish : bool, default=False
        Run full CD after each smooth candidate, allowing deleted coordinates to
        reenter.
    threshold : float, default=0.001
        Nonnegative coefficient-deletion threshold in working units for method="smooth". 
        Deleted coordinates cannot reenter unless polish=True.
    lqa_epsilon : float, default=1e-06
        Positive numerical offset used in the LQA diagonal approximation,
        diag(|grad P_m| / (|beta| + lqa_epsilon)).
    lambda_init : float or None, default=None
        Positive starting multiplier for automatic grids. None uses
        1e-6/(n_samples*max(m,1)); ignored for explicit lambdas.
    lambda_max : float or None, default=None
        Nonnegative endpoint for automatic grids. None uses the endpoint
        heuristic described in path_fridge (unit weights and beta=0.001 for m>0). For m>0 this is not a guaranteed
        support-size bound. Ignored for explicit lambdas.

    Returns
    -------
    result : FridgeCV
        Cross-validation results, selected hyperparameters, and selected
        full-data fit.

    Notes
    -----
    For m > 0, the objective is generally nonconvex, so global optimality is not
    guaranteed. Inspect KKT residuals and convergence flags. Smooth-path solutions
    are approximate unless polished by coordinate descent.

    Preprocessing and adaptive weights are estimated separately within each
    training fold. Explicit weights are held fixed across folds and should therefore
    be specified independently of the validation data.

    Automatic lambda grids are constructed using full-data working scales. The
    global one-SE threshold is the minimum mean fold MSE plus its standard error.
    Automatic final refits use a finer grid; approximate smooth solutions can
    therefore differ from fits traced on the CV grid at the same lambda.
    """

    # Validate independent data arrays and candidate sizes before allocating folds.
    X, y = data_arrays(X, y)
    n_folds = positive_int(n_folds, "n_folds")
    if not 2 <= n_folds <= len(y):
        raise ValueError("n_folds must be between 2 and n_samples")

    # Sort unique m values so the one-SE rule can choose the smallest admissible target size.
    ms = sorted(set(validate_m(m, X.shape[1]) for m in ms))
    if not ms:
        raise ValueError("ms must be nonempty")
    if selection not in {"min", "1se"}:
        raise ValueError("selection must be 'min' or '1se'")
    if method not in {"smooth", "cd"}:
        raise ValueError("method must be 'smooth' or 'cd'")

    # Accept only options that can be applied consistently to every training-fold path.
    # Allow the same smooth-path options for CV folds and the full-data fit.
    allowed = {"weights", "adaptive", "gamma", "fit_intercept", "standardize", "scale_y",
               "tau", "tol", "max_iter", "backend", "block_size", "ridge_init", "weight_epsilon", "warn",
               "correction_steps", "initial_correction_steps", "polish", "threshold",
               "lqa_epsilon", "lambda_init", "lambda_max"}
    unknown = set(fit_options) - allowed
    if unknown:
        raise ValueError(f"Unknown CV options: {sorted(unknown)}")
    
    # Set ridge-initialization default for both fold and full fits.
    fit_options.setdefault("ridge_init", 1e-6)

    # Shuffle indices reproducibly and split them into disjoint folds of nearly equal size.
    shuffled = np.random.default_rng(random_state).permutation(len(y))
    folds = np.array_split(shuffled, n_folds)

    # Store per-model fold diagnostics separately from cached full-data paths.
    results = {}
    full_paths = {}
    best_m, best_index, best_mse, best_se = None, None, np.inf, None

    # Evaluate a shared lambda grid for each m; automatic grids may differ between m values.
    for m in ms:
        # An automatic grid is obtained from a complete full-data path and reused in
        # every training fold. Cache that path for the eventual selected-model refit.
        if lambdas is None:
            baseline = path_fridge(X, y, m, method=method, n_steps=n_steps,
                                   stop_at_target=False, **fit_options)
            full_paths[m] = baseline
            grid = baseline.lambdas_
        else:
            grid = np.asarray(lambdas, dtype=float)

        # Collect fold MSE and stationarity diagnostics independently: failure to
        # meet KKT tolerance is recorded, not substituted for a validation error.
        errors = []
        convergence, stationarity = [], []

        # 1. Exclude the held-out indices. Each path learns scaling and adaptive weights
        # from its training rows, then predicts held-out responses in original units.
        for test in folds:
            train = np.ones(len(y), dtype=bool)
            train[test] = False
            path = path_fridge(X[train], y[train], m, grid, method=method, **fit_options)

            # Broadcast y across path columns to compute a separate held-out MSE for every lambda.
            # ##### IMPROVEMENT: Evaluate each distinct consecutive fit once.
            # Expand repeated predictions back to the full common lambda grid;
            # every fold still contributes to every mean and standard error.
            changed = np.r_[True, np.any(path.coef_path_[1:] != path.coef_path_[:-1], axis=1)
                            | (path.intercepts_[1:] != path.intercepts_[:-1])]
            predictions = path.predict(X[test], index=np.flatnonzero(changed))
            unique_mse = np.mean((predictions - y[test, None]) ** 2, axis=0)
            errors.append(unique_mse[np.cumsum(changed) - 1])
            convergence.append(path.converged_)
            stationarity.append(path.kkt_residuals_)

        # 2. Summarize folds with equal fold weight; standard errors use the sample
        # standard deviation of fold MSE divided by sqrt(n_folds).
        errors = np.asarray(errors)
        mean = errors.mean(axis=0)
        se = errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
        results[m] = {"lambdas": grid.copy(), "fold_mse": errors,
                      "mean_mse": mean, "se_mse": se,
                      "converged": np.asarray(convergence),
                      "kkt_residuals": np.asarray(stationarity)}

        # Compare the best lambda for this m against the best candidate seen so far.
        index = int(np.argmin(mean))
        if mean[index] < best_mse:
            best_m, best_index, best_mse, best_se = m, index, float(mean[index]), float(se[index])

    # 3. Use one global threshold: minimum mean MSE plus the SE at that minimizer.
    # Choose the first admissible m, then its largest admissible lambda.
    threshold = best_mse + best_se
    one_m = next(m for m in ms if np.any(results[m]["mean_mse"] <= threshold))
    admissible = np.flatnonzero(results[one_m]["mean_mse"] <= threshold)
    one_index = admissible[np.argmax(results[one_m]["lambdas"][admissible])]
    best_lambda = float(results[best_m]["lambdas"][best_index])
    one_lambda = float(results[one_m]["lambdas"][one_index])

    # Both selection rules are reported; this choice determines the returned fitted model.
    chosen_m, chosen_lambda = (best_m, best_lambda) if selection == "min" else (one_m, one_lambda)
    
    # Refit automatic searches on 100 intervals. Include the CV-selected lambda 
    # exactly rather than substituting a nearby grid value. Different continuation 
    # spacing can change the approximate full-data fit.
    if lambdas is None:
        cv_grid = results[chosen_m]["lambdas"]
        refit_grid = (np.geomspace(cv_grid[0], cv_grid[-1], 101)
                      if len(cv_grid) > 1 else cv_grid.copy())
        refit_grid = np.unique(np.append(refit_grid, chosen_lambda))
        full_path = path_fridge(X, y, chosen_m, refit_grid, method=method, **fit_options)
    else:
        # Preserve explicitly requested grids without densifying.
        full_path = path_fridge(X, y, chosen_m, results[chosen_m]["lambdas"],
                                method=method, **fit_options)
    chosen_index = int(np.flatnonzero(full_path.lambdas_ == chosen_lambda)[0])
    fit = FridgeResult(
        coef_=full_path.coef_path_[chosen_index].copy(),
        intercept_=float(full_path.intercepts_[chosen_index]),
        scaled_coef_=full_path.scaled_coef_path_[chosen_index].copy(),
        weights_=full_path.weights_.copy(),
        m=chosen_m,
        lam=chosen_lambda,
        tau=float(fit_options.get("tau", 0.0)),
        n_iter_=int(full_path.n_iter_[chosen_index]),
        converged_=bool(full_path.converged_[chosen_index]),
        kkt_residual_=float(full_path.kkt_residuals_[chosen_index]),
        objective_history_=np.array([full_path.objectives_[chosen_index]]),
        backend_=full_path.backend_,
        scaling_=full_path.scaling_,
    )
    return FridgeCV(
        results_=results,
        best_m_=best_m,
        best_lambda_=best_lambda,
        one_se_m_=one_m,
        one_se_lambda_=one_lambda,
        one_se_threshold_=threshold,
        selection=selection,
        fit_=fit,
        method=method,
        path_=full_path,
    )
