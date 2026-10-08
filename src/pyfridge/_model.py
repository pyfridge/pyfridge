"""Estimator-style interface without a mandatory scikit-learn dependency."""

import numpy as np

from ._coordinate import fit_fridge


class Fridge:
    """Estimator interface for a single coordinate-descent Fridge fit.

    Parameters
    ----------
    m : int, default=1
        Target model size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    lam : float, default=0.1
        Nonnegative penalty multiplier in the working-space objective RSS/(2n) +
        lam*P_m + tau*||beta||^2/2.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative penalty weights in working coefficient units. Explicit weights
        override adaptive weighting; None uses ones unless adaptive=True.
    adaptive : bool, default=True
        Estimate weights from a ridge pilot when weights is None.
    gamma : float, default=1.0
        Nonnegative adaptive-weight exponent:
        (abs(beta_ridge)+weight_epsilon)**(-gamma).
    beta_init : array_like of shape (n_features,) or None, default=None
        Initial coefficients in working, centered/scaled units. None uses ridge
        initialization. The solver copies this vector before updating coefficients.
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
        Maximum positive number of CD sweeps during fitting.
    zero_tol : float, default=0.0
        Nonnegative cutoff that sets small coordinate updates to zero. Positive
        values may prevent satisfying the KKT tolerance.
    sorted_cd : bool, default=True
        Visit coordinates in descending absolute coefficient order at each sweep;
        all coordinates, including zeros, are revisited.
    backend : {"auto", "python", "cython"}, default='auto'
        CD implementation. Auto uses the optional compiled extension when available;
        cython requires it. Unpolished smooth paths use NumPy/SciPy.
    block_size : int or None, default=None
        Positive CD block size for caching penalty calculations. None uses
        max(1, int(sqrt(n_features))).
    ridge_init : float, default=0.05
        Nonnegative normalized ridge strength for initialization and adaptive-weight
        pilots. Zero uses least squares.
    weight_epsilon : float, default=1e-05
        Positive offset in adaptive weights, preventing division by zero.
    warn : bool, default=True
        Emit RuntimeWarning when the fit fails the KKT convergence check.

    Attributes
    ----------
    fit_result_ : FridgeResult
        Complete fit and diagnostics, available after fit.
    coef_ : ndarray of shape (n_features,)
        Fitted original-unit coefficients.
    intercept_ : float
        Fitted original-unit intercept.
    scaling_ : Scaling
        Preprocessing means, scales, and zero-column mask. This is the same
        object as fit_result_.scaling_, available after fitting.
    n_features_in_ : int
        Number of fitted predictors.

    Notes
    -----
    Other fitted diagnostics mirror fit_result_. Constructor values are stored
    unvalidated until fit; for m>0 global optimality is not guaranteed.
    """
    def __init__(self, m=1, lam=0.1, *, weights=None, adaptive=True, gamma=1.0,
                 beta_init=None, fit_intercept=True, standardize=True, scale_y=False,
                 tau=0.0, tol=1e-7, max_iter=10000, zero_tol=0.0, sorted_cd=True,
                 backend="auto", block_size=None, ridge_init=0.05, weight_epsilon=1e-5,
                 warn=True):
        """Initialize the Fridge estimator.

        Parameters
        ----------
        m : int, default=1
            Target model size, between 0 and n_features. The penalty has degree m+1; m=0
            is weighted Lasso. This is not a hard support constraint.
        lam : float, default=0.1
            Nonnegative penalty multiplier in the working-space objective RSS/(2n) +
            lam*P_m + tau*||beta||^2/2.
        weights : array_like of shape (n_features,) or None, default=None
            Nonnegative penalty weights in working coefficient units. Explicit weights
            override adaptive weighting; None uses ones unless adaptive=True.
        adaptive : bool, default=True
            Estimate weights from a ridge pilot when weights is None.
        gamma : float, default=1.0
            Nonnegative adaptive-weight exponent:
            (abs(beta_ridge)+weight_epsilon)**(-gamma).
        beta_init : array_like of shape (n_features,) or None, default=None
            Initial coefficients in working, centered/scaled units. None uses ridge
            initialization. The solver copies this vector before updating coefficients.
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
            Maximum positive number of CD sweeps during fitting.
        zero_tol : float, default=0.0
            Nonnegative cutoff that sets small coordinate updates to zero. Positive
            values may prevent satisfying the KKT tolerance.
        sorted_cd : bool, default=True
            Visit coordinates in descending absolute coefficient order at each sweep;
            all coordinates, including zeros, are revisited.
        backend : {"auto", "python", "cython"}, default='auto'
            CD implementation. Auto uses the optional compiled extension when available;
            cython requires it. Unpolished smooth paths use NumPy/SciPy.
        block_size : int or None, default=None
            Positive CD block size for caching penalty calculations. None uses
            max(1, int(sqrt(n_features))).
        ridge_init : float, default=0.05
            Nonnegative normalized ridge strength for initialization and adaptive-weight
            pilots. Zero uses least squares.
        weight_epsilon : float, default=1e-05
            Positive offset in adaptive weights, preventing division by zero.
        warn : bool, default=True
            Emit RuntimeWarning when the fit fails the KKT convergence check.
        """
        # Store constructor settings without fitting. Numerical validation occurs in fit_fridge.
        self.m = m
        self.lam = lam
        self.weights = weights
        self.adaptive = adaptive
        self.gamma = gamma

        self.beta_init = beta_init
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.scale_y = scale_y
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

        self.zero_tol = zero_tol
        self.sorted_cd = sorted_cd
        self.backend = backend
        self.block_size = block_size
        self.ridge_init = ridge_init
        self.weight_epsilon = weight_epsilon
        self.warn = warn

    def fit(self, X, y):
        """Fit the estimator and store coefficients and diagnostics.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Finite design matrix.
        y : array_like of shape (n_samples,)
            Finite response vector.

        Returns
        -------
        self : Fridge
            Fitted estimator, with fit_result_, coef_, intercept_, scaling_, and solver diagnostics.
        """

        # Pass constructor settings to the functional solver, then expose its
        # original-unit coefficients and diagnostic fields on this estimator.
        self.fit_result_ = fit_fridge(X, y, **self.get_params())
        
        for name in ("coef_", "intercept_", "scaled_coef_", "weights_", "n_iter_",
                     "converged_", "kkt_residual_", "objective_history_", "backend_", "scaling_"):
            setattr(self, name, getattr(self.fit_result_, name))
        self.n_features_in_ = self.coef_.size
        return self

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

        # Require fitted state before delegating to the result object.
        if not hasattr(self, "fit_result_"):
            raise ValueError("Call fit before predict")
        return self.fit_result_.predict(X)

    def score(self, X, y):
        """Compute the coefficient of determination (R-squared).

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Finite design matrix.
        y : array_like of shape (n_samples,)
            Finite response vector.

        Returns
        -------
        score : float
            R-squared; it can be negative. For constant y, returns 1 for an exact fit and 0 otherwise.
        """
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        if y.shape != pred.shape or not np.isfinite(y).all():
            raise ValueError("y must match the number of predictions and be finite")

        # Compute residual and total sums of squares in original response units.
        residual = float(np.sum((y - pred) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))

        # For constant responses, define score as one for zero error and zero otherwise.
        return 1.0 - residual / total if total > 0 else float(residual == 0)

    def get_params(self, deep=True):
        """Return estimator constructor parameters.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for estimator compatibility. There are no nested estimators, so it does
            not alter the result.

        Returns
        -------
        params : dict
            Constructor parameter names and current values; excludes fitted state.
        """

        # Expose only constructor parameters, never learned coefficients or diagnostics.
        # These include all solver controls.
        names = ("m", "lam", "weights", "adaptive", "gamma", "beta_init", "fit_intercept",
                 "standardize", "scale_y", "tau", "tol", "max_iter", "zero_tol", "sorted_cd",
                 "backend", "block_size", "ridge_init", "weight_epsilon", "warn")
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        """Update estimator constructor parameters.

        Parameters
        ----------
        params : keyword arguments
            Constructor parameter names and replacement values. Unknown names raise
            ValueError.

        Returns
        -------
        self : Fridge
            Estimator with updated parameters. Call fit again to apply them to fitted results.
        """

        # Validate each requested name before assigning its new constructor value.
        # Updating parameters does not recompute an existing fit; fit must be called again.
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Unknown Fridge parameter: {name}")
            setattr(self, name, value)
        return self
