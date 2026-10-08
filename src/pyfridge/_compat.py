"""Adapters for the names used in previous version of the script."""

import numpy as np

from ._coordinate import fit_fridge
from ._path import path_fridge, refit_ols
from ._penalty import penalty_derivatives



def lm_fridge_cd(x, y, m, lam, betainit=None, penalty_factor=None, tau=0.0,
                 thres=0.0, tol=1e-7, max_iter=10000, scaled_input=False,
                 sorted_cd=True, verbose=False, *, backend="auto", block_size=None):
    """Fit coordinate descent using the legacy dictionary interface.

    Parameters
    ----------
    x : array_like of shape (n_samples, n_features)
        Finite design matrix; columns are predictors.
    y : array_like of shape (n_samples,)
        Finite response vector.
    m : int
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    lam : float
        Nonnegative Fridge penalty multiplier in the normalized RSS/(2n) objective.
    betainit : array_like of shape (n_features,) or None, default=None
        Initial coefficients in working units. None uses a ridge pilot.
    penalty_factor : array_like of shape (n_features,) or None, default=None
        Nonnegative penalty weights; None uses ones.
    tau : float, default=0.0
        Nonnegative ridge multiplier in tau*||beta||^2/2; this is separate from lam.
    thres : float, default=0.0
        Nonnegative working-unit coefficient cutoff. Zero disables truncation.
    tol : float, default=1e-07
        Positive KKT tolerance, scaled by max(1, max(abs(X.T@y/n))) in working units.
        Smooth corrections also use this as a stopping tolerance.
    max_iter : int, default=10000
        Maximum positive number of CD sweeps per fit or path point; used by method="cd"
        or polishing.
    scaled_input : bool, default=False
        True assumes already preprocessed inputs and fixes intercept to zero. False
        centers and population-standardizes both x and y.
    sorted_cd : bool, default=True
        Visit coordinates in descending absolute coefficient order at each sweep; all
        coordinates are revisited.
    verbose : bool, default=False
        Print sweeps, convergence status, and KKT residual.
    backend : {"auto", "python", "cython"}, default='auto'
        CD implementation. Auto uses the optional compiled extension when available;
        cython requires it. Unpolished smooth paths use NumPy/SciPy.
    block_size : int or None, default=None
        Positive CD block size for caching penalty calculations. None uses max(1,
        int(sqrt(n_features))).

    Returns
    -------
    result : dict
        Keys: coef (intercept followed by original-unit coefficients), intercept, scaled_coef, n_iter, penalty.factor (weights), converged, kkt_residual, m, lam, tau, and scaler. The scaler contains xm, xsd, ym, and ysd.
    """

    # Translate legacy names to the normalized solver. Already-scaled input
    # disables preprocessing; otherwise both predictors and response are standardized.
    fit = fit_fridge(x, y, m, lam, weights=penalty_factor, beta_init=betainit,
                     tau=tau, zero_tol=thres, tol=tol, max_iter=max_iter,
                     sorted_cd=sorted_cd, fit_intercept=not scaled_input,
                     standardize=not scaled_input, scale_y=not scaled_input,
                     backend=backend, block_size=block_size)

    # Print diagnostics only on explicit request; return the legacy dictionary in either case.
    if verbose:
        print(f"CD sweeps={fit.n_iter_}, converged={fit.converged_}, KKT={fit.kkt_residual_:.3g}")
    return fit.as_dict()


def _fridge_path_nlambda(X, y, m, weights, adaptive, gamma, n_steps, thres,
                   lam_init, correction_steps):

    # Notebook multipliers belong to the unnormalized RSS/2 objective. Divide by n
    # when calling the normalized package solver, then multiply the returned grid by n.
    n = len(y)
    initial = 1e-6 / max(m, 1) if lam_init is None else lam_init
    path = path_fridge(X, y, m, weights=weights, adaptive=adaptive, gamma=gamma,
                       n_steps=n_steps, threshold=thres,
                       lambda_init=initial / n, ridge_init=initial / n,
                       fit_intercept=False, standardize=False, scale_y=False,
                       correction_steps=correction_steps)
    return path.lambdas_ * n, path.coef_path_


def adafridge_solution_path(X, y, m=1, weights=None, gamma=1.0,
                            n_steps=100, thres=1e-3, lam_init=None,
                            *, correction_steps=2):
    """Compute an adaptive smooth path for already preprocessed notebook data.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix. Inputs are copied; columns are predictors.
    y : array_like of shape (n_samples,)
        Finite response vector.
    m : int, default=1
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative working-unit weights. None estimates adaptive ridge weights;
        an all-ones vector explicitly requests an unweighted penalty.
    gamma : float, default=1.0
        Nonnegative adaptive-weight exponent:
        (abs(beta_ridge)+weight_epsilon)**(-gamma).
    n_steps : int, default=100
        Positive number of intervals in an automatic grid, normally giving n_steps+1
        points. Degenerate grids may contain one point; smooth stopping can shorten
        paths. Ignored for explicit lambdas.
    thres : float, default=0.001
        Nonnegative working-unit coefficient cutoff. Zero disables truncation.
    lam_init : float or None, default=None
        Positive initial unnormalized notebook multiplier. None uses 1e-6/max(m,1).
        Internally divided by n_samples for path_fridge and ridge initialization.
    correction_steps : int, default=2
        Maximum positive number of repeated LQA corrections after each smooth predictor
        step; early stopping may use fewer.

    Returns
    -------
    lambdas : ndarray of shape (n_lambdas,)
        Unnormalized notebook multipliers, equal to n_samples times package lambdas.
    coefficients : ndarray of shape (n_lambdas, n_features)
        Coefficients in supplied working units. Automatic paths may stop at support size <=m. Both predictor and corrections use LQA; no preprocessing is performed.
    """

    # Delegate to the shared notebook path adapter without preprocessing supplied arrays.
    return _fridge_path_nlambda(X, y, m, weights, True, gamma, n_steps, thres,
                          lam_init, correction_steps)


def differential_solution_path(X, y, m=1, n_steps=100, thres=1e-3,
                               lam_init=None, *, correction_steps=2):
    """Compute an unweighted smooth path for already preprocessed notebook data.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix. Inputs are copied; columns are predictors.
    y : array_like of shape (n_samples,)
        Finite response vector.
    m : int, default=1
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    n_steps : int, default=100
        Positive number of intervals in an automatic grid, normally giving n_steps+1
        points. Degenerate grids may contain one point; smooth stopping can shorten
        paths. Ignored for explicit lambdas.
    thres : float, default=0.001
        Nonnegative working-unit coefficient cutoff. Zero disables truncation.
    lam_init : float or None, default=None
        Positive initial unnormalized notebook multiplier. None uses 1e-6/max(m,1).
        Internally divided by n_samples for path_fridge and ridge initialization.
    correction_steps : int, default=2
        Maximum positive number of repeated LQA corrections after each smooth predictor
        step; early stopping may use fewer.

    Returns
    -------
    lambdas : ndarray of shape (n_lambdas,)
        Unnormalized notebook multipliers, equal to n_samples times package lambdas.
    coefficients : ndarray of shape (n_lambdas, n_features)
        Coefficients in supplied working units. Automatic paths may stop at support size <=m. Both predictor and corrections use LQA; no preprocessing is performed.
    """

    # Delegate to the shared notebook path adapter without preprocessing supplied arrays.
    return _fridge_path_nlambda(X, y, m, None, False, 1.0, n_steps, thres,
                          lam_init, correction_steps)


def gh_adaptive_fridge(beta, m, weights=None, compute_hessian=True):
    """Compute weighted penalty derivatives using the notebook interface.

    Parameters
    ----------
    beta : array_like of shape (n_features,)
        Finite, nonempty coefficient vector.
    m : int
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative weights applied to abs(beta). None uses ones.
    compute_hessian : bool, default=True
        Return the analytical orthant Hessian when True; return None as the second
        output when False. Smooth path algorithms always use LQA instead.

    Returns
    -------
    gradient : ndarray of shape (n_features,)
        Gradient on nonzero orthants; zero coefficients receive the selected subgradient sign(0)=0.
    hessian : ndarray of shape (n_features, n_features) or None
        Analytical Hessian on smooth orthants, or None when compute_hessian=False. At zero coefficients it is not a classical Hessian.
    """

    # Reuse the shared mathematical derivative implementation rather than duplicating it.
    return penalty_derivatives(beta, m, weights, compute_hessian)


def fridge_penalty_derivatives(beta, m, compute_hessian=True):
    """Compute unweighted penalty derivatives using the notebook interface.

    Parameters
    ----------
    beta : array_like of shape (n_features,)
        Finite, nonempty coefficient vector.
    m : int
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    compute_hessian : bool, default=True
        Return the analytical orthant Hessian when True; return None as the second
        output when False. Smooth path algorithms always use LQA instead.

    Returns
    -------
    gradient : ndarray of shape (n_features,)
        Gradient on nonzero orthants; zero coefficients receive the selected subgradient sign(0)=0.
    hessian : ndarray of shape (n_features, n_features) or None
        Analytical Hessian on smooth orthants, or None when compute_hessian=False. At zero coefficients it is not a classical Hessian.
    """

    # Reuse the shared mathematical derivative implementation rather than duplicating it.
    return penalty_derivatives(beta, m, compute_hessian=compute_hessian)


def final_ols(X, y, beta_final):
    """Refit OLS without an intercept on a notebook coefficient support.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Finite design matrix. Inputs are copied; columns are predictors.
    y : array_like of shape (n_samples,)
        Finite response vector.
    beta_final : array_like of shape (n_features,)
        Coefficient vector whose nonzero entries specify the OLS refit support.

    Returns
    -------
    coef : ndarray of shape (n_features,)
        OLS coefficients, zero outside the supplied support; no intercept is fitted.
    """

    # Return only coefficients for legacy callers; this adapter fits no intercept.
    return refit_ols(X, y, beta_final, fit_intercept=False)[0]


def predict_fridgec(fit_fridge, newx):
    """Predict from a legacy fit dictionary.

    Parameters
    ----------
    fit_fridge : dict
        Legacy fit dictionary containing "coef", with intercept first followed by
        n_features coefficients.
    newx : array_like of shape (n_samples, n_features)
        Finite new design matrix in original input units.

    Returns
    -------
    prediction : ndarray of shape (n_samples,)
        Predicted responses in original units.
    """

    # Validate the feature count against a legacy coefficient vector with intercept first.
    newx = np.asarray(newx, dtype=float)
    coef = np.asarray(fit_fridge["coef"], dtype=float)
    if newx.ndim != 2 or coef.shape != (newx.shape[1] + 1,) or not np.isfinite(newx).all():
        raise ValueError("newx must be finite and match the fitted columns")

    # Separate the stored intercept from slope coefficients when forming predictions.
    return newx @ coef[1:] + coef[0]
