# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs


# Reuse residuals and block-exterior polynomials across coordinate updates.
def sweep(double[::1, :] X, double[::1] y, double[::1] beta,
          double[::1] residual, double[::1] weights, double[::1] curvature,
          cnp.intp_t[::1] order, int m, double lam, double tau,
          double zero_tol, int block_size):
    """
    One full sweep; callers validate all arrays before entering this kernel.
    The sweep updates `beta` and residual in-place. Only returns the amount 
    of largest coefficient update (`max_change`).
    """

    # Typed memoryviews and C scalar declarations keep inner numerical loops native.
    # X has contiguous columns; beta and residual are modified in place.
    cdef Py_ssize_t n = X.shape[0], p = X.shape[1]
    cdef Py_ssize_t start, end, pos, pos2, j, h, i, k
    cdef double z, value, factor, old, new, delta, max_change = 0

    # Allocate reusable degree-m polynomial buffers and a mask for the current block.
    cdef double[::1] exterior = np.zeros(m + 1)
    cdef double[::1] interior = np.zeros(m + 1)
    cdef cnp.uint8_t[::1] in_block = np.zeros(p, dtype=np.uint8)

    # Visit blocks in the supplied coordinate order; outside coefficients stay fixed during each block.
    for start in range(0, p, block_size):
        end = min(start + block_size, p)

        # Mark the current block so the exterior recurrence excludes these coordinates.
        for pos in range(start, end):
            in_block[order[pos]] = 1

        # Reset the exterior product sums, including the empty product P_{-1}=1, before processing this block.
        for k in range(m + 1):
            exterior[k] = 0
        exterior[0] = 1

        # Accumulate products outside the block only when a nonconstant polynomial is needed.
        if m > 0 and lam > 0:
            for h in range(p):
                if in_block[h] or beta[h] == 0 or weights[h] == 0:
                    continue
                value = fabs(beta[h]) * weights[h]

                # Update degrees in descending order so every recurrence uses the previous state.
                for k in range(m, 0, -1):
                    exterior[k] += value * exterior[k - 1]

        # Minimize each conditional objective using the most recent in-place coefficients.
        for pos in range(start, end):
            j = order[pos]
            old = beta[j]
            factor = 0
            if lam > 0:
                for k in range(m + 1):
                    interior[k] = 0
                interior[0] = 1

                # Rebuild the interior polynomial for this coordinate, excluding the coordinate itself.
                if m > 0:
                    for pos2 in range(start, end):
                        h = order[pos2]
                        if h == j or beta[h] == 0 or weights[h] == 0:
                            continue
                        value = fabs(beta[h]) * weights[h]
                        for k in range(m, 0, -1):
                            interior[k] += value * interior[k - 1]

                # Convolve exterior and interior coefficients to obtain the leave-one-out degree-m term.
                for k in range(m + 1):
                    factor += exterior[k] * interior[m - k]
                # Chain rule multiplies the leave-one-out polynomial by weights[j].
                factor *= lam * weights[j]

            # Compute the current residual correlation, then add back the old coordinate data contribution.
            z = 0
            for i in range(n):
                z += X[i, j] * residual[i]
            
            # Include actual predictor curvature; tau belongs in the denominator.
            z = z / n + old * (curvature[j] - tau)

            # Soft-threshold the partial score and divide by curvature. A zero-curvature
            # coordinate is set to zero; the ridge term is already included in curvature.
            if curvature[j] == 0 or fabs(z) <= factor:
                new = 0
            elif z > 0:
                new = (z - factor) / curvature[j]
            else:
                new = (z + factor) / curvature[j]

            # Apply optional truncation before recording the actual coefficient displacement.
            if fabs(new) < zero_tol:
                new = 0
            delta = new - old
            beta[j] = new
            if delta != 0:
                # Update residuals after every change, including truncation to zero.
                for i in range(n):
                    residual[i] -= X[i, j] * delta
                max_change = max(max_change, fabs(delta))

        # Clear the block membership mask before selecting the next block.
        for pos in range(start, end):
            in_block[order[pos]] = 0

    return max_change
