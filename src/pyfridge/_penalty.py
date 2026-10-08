"""Stable Fridge penalty P_m and derivatives."""

import numpy as np

from ._data import validate_m, vector


# Positive dynamic programming replaces the subtraction-based recurrence.
def _P_minus_one(z, degree):
    if degree == 0:
        return np.ones(1)

    # Store the product sums of degrees 0,...,degree; entry k equals P_{k-1}(z).
    # Initialize the degree-zero entry to one (the empty product, P_{-1}(z)=1).
    pk = np.zeros(degree + 1)
    pk[0] = 1.0
    count = 0

    # Add one weighted absolute coefficient at a time to the polynomial.
    # Zero entries contribute no new products and can be skipped.
    for value in z:
        if value == 0:
            continue
        count += 1

        # No product can have degree larger than the number of nonzero entries seen.
        k = min(count, degree)

        # Apply P_{k-1}(new)=P_{k-1}(old)+value*P_{k-2}(old). The copy prevents
        # newly updated values from leaking into the same recurrence step.
        pk[1:k + 1] += value * pk[:k].copy()
    return pk


def _loo(z, degree):
    p = len(z)
    if degree == 0:
        return np.ones(p)

    # Excluding one of p entries leaves p-1 entries, so degree >=p is impossible.
    if degree >= p:
        return np.zeros(p)

    # Build polynomial tables for every prefix and suffix. Combining the tables
    # lets us exclude each coordinate without subtracting nearly equal quantities.
    prefix = np.zeros((p + 1, degree + 1))
    suffix = np.zeros_like(prefix)
    prefix[:, 0] = suffix[:, 0] = 1.0

    # prefix[i] contains products from entries strictly before i.
    for i in range(p):
        prefix[i + 1, 1:] = prefix[i, 1:] + z[i] * prefix[i, :-1]

    # suffix[i+1] contains products from entries strictly after i.
    for i in range(p - 1, -1, -1):
        suffix[i, 1:] = suffix[i + 1, 1:] + z[i] * suffix[i + 1, :-1]

    # Reverse the degree axis and convolve to extract degree exactly equal to degree.
    return np.sum(prefix[:-1] * suffix[1:, ::-1], axis=1)


def penalty(beta, m=1, weights=None):
    """Evaluate the weighted Fridge penalty P_m.

    Parameters
    ----------
    beta : array_like of shape (n_features,)
        Finite, nonempty coefficient vector.
    m : int, default=1
        Target support size, between 0 and n_features. The penalty has degree m+1; m=0
        is weighted Lasso. This is not a hard support constraint.
    weights : array_like of shape (n_features,) or None, default=None
        Nonnegative weights applied to abs(beta). None uses ones.

    Returns
    -------
    value : float
        P_m(beta), the sum over all subsets of m+1 distinct coordinates of
        the products of weights[j]*abs(beta[j]). At m=0 this is weighted L1;
        at m=n_features it is zero.
    """

    # Validate a finite coefficient vector and the degree before polynomial calculations.
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1 or beta.size == 0:
        raise ValueError("beta must be a nonempty vector")
    
    beta = vector(beta, beta.size, "beta")
    m = validate_m(m, beta.size)
    w = np.ones(beta.size) if weights is None else vector(
        weights, beta.size, "weights", nonnegative_only=True
    )
    
    if m == beta.size:
        return 0.0

    # Convert polynomial overflow into a clear exception instead of returning an unusable penalty.
    with np.errstate(over="raise", invalid="raise"):
        try:
            return float(_P_minus_one(w * np.abs(beta), m + 1)[m + 1])
        except FloatingPointError as exc:
            raise ValueError("penalty overflow; rescale data/weights or reduce m") from exc


def threshold_factors(beta, m, weights):
    # The derivative with respect to |beta_j| is w_j times the degree-m
    # polynomial on all other weighted absolute coefficients.
    return weights * _loo(weights * np.abs(beta), m)


# Hessians apply on smooth orthants; zeros use sign(0)=0 as a chosen subgradient.
def penalty_derivatives(beta, m=1, weights=None, compute_hessian=True):
    """Compute a subgradient of P_m and its optional analytical orthant Hessian.

    Parameters
    ----------
    beta : array_like of shape (n_features,)
        Finite, nonempty coefficient vector.
    m : int, default=1
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
        Analytical Hessian on smooth orthants, or None when compute_hessian=False. At zero 
        coefficients, the returned matrix uses sign(0)=0 in the orthant formula and is not
        a classical Hessian.
    """

    # Validate a finite coefficient vector and the degree before polynomial calculations.
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1 or beta.size == 0:
        raise ValueError("beta must be a nonempty vector")
    beta = vector(beta, beta.size, "beta")
    p = beta.size
    m = validate_m(m, p)
    w = np.ones(p) if weights is None else vector(weights, p, "weights", nonnegative_only=True)

    # Separate absolute magnitudes from signed chain-rule weights for orthant derivatives.
    z = w * np.abs(beta)
    signed_w = np.sign(beta) * w

    # Each gradient entry excludes its own coordinate from the degree-m product.
    grad = signed_w * _loo(z, m)
    if not compute_hessian:
        return grad, None

    # Within an orthant the penalty is linear in each individual coordinate, so
    # the Hessian diagonal is zero. Off-diagonal entries exclude both coordinates.
    hess = np.zeros((p, p))
    if 0 < m < p:
        for i in range(p):
            # For row i, first remove i; leave-one-out products then also remove each column j.
            mask = np.arange(p) != i
            hess[i, mask] = signed_w[i] * signed_w[mask] * _loo(z[mask], m - 1)
    return grad, hess
