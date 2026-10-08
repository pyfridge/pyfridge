"""Validation and preprocessing shared by both solvers."""

# Centralize validation/scaling instead of repeating notebook assumptions.
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import solve


# Reject invalid inputs before entering unchecked Cython loops.
def data_arrays(X, y):
    # Allocate independent float64 arrays. Fortran-order X gives contiguous
    # predictor columns to the coordinate-descent kernels.
    X = np.array(X, dtype=np.float64, order="F", copy=True)
    y = np.array(y, dtype=np.float64, copy=True)

    # Reject shape mismatches before downstream kernels are called.
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2-dimensional and y must be 1-dimensional")
    if X.shape[0] != y.size or min(X.shape) < 1:
        raise ValueError("X and y need matching, nonempty observations and predictors")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite numbers")
    return X, y


# m: target model size. penalty's polynomial degree is m+1.
def validate_m(m, p):
    # Reject booleans explicitly: Python treats bool as an integer, but it is not a model size.
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or not 0 <= m <= p:
        raise ValueError(f"m must be an integer between 0 and {p}")
    return int(m)


# Validate every regularization/tolerance parameter consistently.
def nonnegative(value, name, *, positive=False):
    # Normalize scalar controls to floating point, then enforce finiteness and sign.
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


# Floating-point iteration counts used to fail later inside range().
def positive_int(value, name):
    # Require a genuine positive integer for loop counts and block sizes.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


# Copy coefficients/weights input so warm starts cannot mutate caller arrays.
def vector(value, p, name, *, nonnegative_only=False):
    # Copy vectors because weights and warm starts must not alias global variables.
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (p,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {p}")
    if nonnegative_only and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass
class Scaling:
    # Record preprocessing so all coefficients and predictions use original units.
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    constant: np.ndarray

    # Map either one fit or a whole path back to original data units.
    def restore(self, beta):
        # Undo response and predictor scaling for either one fit or multiple coefficient rows.
        coef = np.asarray(beta) * self.y_scale / self.x_scale

        # Undo centering: intercept = response mean - original coefficients @ predictor means.
        intercept = self.y_mean - coef @ self.x_mean
        return coef, intercept


# Use population scaling and set coefficients for constant predictors/response to 0.
def prepare(X, y, fit_intercept, standardize, scale_y):
    # Work on independent arrays before applying in-place centering and scaling.
    X, y = data_arrays(X, y)
    p = X.shape[1]

    # Center only when an intercept is fitted.
    xm = X.mean(axis=0) if fit_intercept else np.zeros(p)
    ym = float(y.mean()) if fit_intercept else 0.0
    X -= xm
    y -= ym

    # Mark columns that are identically zero in working space so their coefficients can be set to 0.
    constant = np.all(X == 0, axis=0)

    # Use root-mean-square scaling, equivalent to population standard deviation
    # after centering. Give 0-columns scale 1 to avoid division by zero.
    xs = np.sqrt(np.mean(X * X, axis=0)) if standardize else np.ones(p)
    xs[xs == 0] = 1.0

    # Apply the same scaling convention to the response.
    ys = float(np.sqrt(np.mean(y * y))) if scale_y else 1.0
    if ys == 0:
        ys = 1.0

    # Return working arrays together with all metadata needed to reverse centering/scaling.
    X /= xs
    y /= ys
    return X, y, Scaling(xm, xs, ym, ys, constant)


# Use an n-by-n dual form of ridge when p exceeds n; avoid a dense p-by-p inverse.
def ridge_start(X, y, alpha):
    n, p = X.shape

    # With no ridge penalty, use the minimum-norm least-squares initializer.
    if alpha == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]

    # For wide designs, solve the dual (X X.T + n*alpha*I), then multiply by X.T.
    if p > n:
        system = X @ X.T
        system.flat[::n + 1] += n * alpha
        return X.T @ solve(system, y, assume_a="pos", check_finite=False)

    # For narrow designs solve the primary ridge problem.
    system = X.T @ X
    system.flat[::p + 1] += n * alpha
    return solve(system, X.T @ y, assume_a="pos", check_finite=False)


# initial ridge to calculate adaptive weights is fitted on the same 
# preprocessed data as the Fridge model.
def resolve_weights(X, y, weights, adaptive, gamma, ridge_init, weight_epsilon):
    p = X.shape[1]

    # check that gamma is nonnegative and weight_epsilon is positive
    gamma = nonnegative(gamma, "gamma")
    weight_epsilon = nonnegative(weight_epsilon, "weight_epsilon", positive=True)

    # If weights are provided, override weight computation.
    if weights is not None:
        return vector(weights, p, "weights", nonnegative_only=True)

    # Unweighted fitting uses weight=1 for every coordinate.
    if not adaptive:
        return np.ones(p)

    # Compute the initial ridge in the same working units used by the final optimization.
    beta = ridge_start(X, y, ridge_init)
    with np.errstate(over="raise", invalid="raise"):
        try:
            # Small initial coefficients receive larger weights. The positive offset
            # keeps weights finite at zero; overflow is converted to an exception.
            w = (np.abs(beta) + weight_epsilon) ** (-gamma)
        except FloatingPointError as exc:
            raise ValueError("adaptive weights overflow; increase weight_epsilon or reduce gamma") from exc
    return vector(w, p, "adaptive weights", nonnegative_only=True)
