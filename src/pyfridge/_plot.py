"""Matplotlib plots of Fridge solution paths and cross-validation results."""

import numpy as np


def _axes(ax):
    # Import only when a plot is requested, keeping fitting dependencies minimal.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requires Matplotlib; install "pyfridge[plot]".') from exc
    return plt.subplots(figsize=(7, 4.5))[1] if ax is None else ax


def _lambda_scale(ax, lambdas, xscale):
    # Retain lambda=0 instead of silently dropping it on a logarithmic axis.
    if xscale not in ("auto", "linear", "log", "symlog"):
        raise ValueError("xscale must be 'auto', 'linear', 'log', or 'symlog'")
    positive = lambdas[lambdas > 0]
    if xscale == "auto":
        xscale = "log" if np.all(lambdas > 0) else "symlog" if positive.size else "linear"
    if xscale == "log" and np.any(lambdas <= 0):
        raise ValueError("A log axis requires strictly positive lambdas; use 'auto' or 'symlog'.")
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=float(positive.min()) if positive.size else 1.0)
    else:
        ax.set_xscale(xscale)


def plot_path_fridge(path, *, feature_names=None, features=None, scaled=True,
                     ax=None, xscale="auto", legend=None):
    """Plot coefficient trajectories against the Fridge penalty parameter.

    Parameters
    ----------
    path : FridgePath
        Result returned by path_fridge, or the path_ attribute of FridgeCV.
    feature_names : sequence of str or None, default=None
        Labels for all predictors, in design-column order. Length must equal
        n_features. None uses x0, x1, and so on.
    features : sequence of int or None, default=None
        Unique predictor indices to display, in the requested order. None shows
        predictors that are nonzero somewhere along the path. An empty sequence
        displays axes without coefficient trajectories.
    scaled : bool, default=True
        Use working-unit coefficients when True; otherwise use original units.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes to draw on. None creates a new figure and axes.
    xscale : {"auto", "linear", "log", "symlog"}, default="auto"
        Lambda-axis scale. Auto uses log for strictly positive grids, symlog
        when zero occurs alongside positive lambdas, and linear for all zeros.
        Explicit log requires all lambdas to be positive.
    legend : bool or None, default=None
        Show predictor labels when True; hide them when False. None shows a
        legend for 1 to 12 displayed predictors to avoid a crowded large legend.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the coefficient trajectories. Use ax.figure.savefig(...)
        to save the figure. This function does not call show or modify the fit.

    Notes
    -----
    Plotting module is optional; install pyfridge[plot] to enable plotting.
    """
    beta = np.asarray(path.scaled_coef_path_ if scaled else path.coef_path_)
    lambdas = np.asarray(path.lambdas_, dtype=float)
    p = beta.shape[1]
    names = [f"x{j}" for j in range(p)] if feature_names is None else list(feature_names)
    if len(names) != p:
        raise ValueError("feature_names must have one label per predictor")

    # Validate requested columns before creating a figure. Default selection
    # excludes trajectories that are zero everywhere, but includes later entrants.
    if features is None:
        indices = np.flatnonzero(np.any(beta != 0, axis=0))
    else:
        indices = np.asarray(features)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("features must be a one-dimensional sequence of integer indices")
        indices = indices.astype(int)
        if np.any(indices < 0) or np.any(indices >= p) or len(np.unique(indices)) != len(indices):
            raise ValueError("features must contain unique indices between 0 and n_features-1")

    ax = _axes(ax)
    _lambda_scale(ax, lambdas, xscale)
    for j in indices:
        ax.plot(lambdas, beta[:, j], linewidth=1.5, label=str(names[j]))
    ax.axhline(0, color="0.5", linewidth=0.7, linestyle=":", label="_zero")
    ax.set(xlabel=r"$\lambda$",
           ylabel="Coefficient (working units)" if scaled else "Coefficient (original units)")
    ax.grid(alpha=0.2)
    if legend is True or (legend is None and 0 < len(indices) <= 12):
        ax.legend(frameon=False)
    return ax


# Mark the global minimum and one-SE choice on their own m curves.
def plot_cv_fridge(cv, *, ax=None, xscale="auto", show_se=True):
    """Plot mean CV errors with the global minimum and one-SE choices.

    Parameters
    ----------
    cv : FridgeCV
        Result returned by cv_fridge. Every candidate m is displayed, so choices
        at different m values are marked on their respective error curves.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes to draw on. None creates a new figure and axes.
    xscale : {"auto", "linear", "log", "symlog"}, default="auto"
        Lambda-axis scale. Auto uses log for strictly positive grids, symlog
        when zero occurs alongside positive lambdas, and linear for all zeros.
        Explicit log requires all lambdas to be positive.
    show_se : bool, default=True
        Shade mean fold MSE plus/minus one standard error around each curve.
        These bands describe fold variability; they are not confidence intervals.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing all CV curves, labeled minimum and one-SE points,
        and the global one-SE threshold. Use ax.figure.savefig(...) to save
        the figure. This function does not call show or rerun cross-validation.

    Notes
    -----
    The threshold is the global minimum mean MSE plus its standard error. The
    one-SE rule chooses the smallest admissible m and then the largest lambda
    below that threshold at that m. MSE uses the original response units.
    Matplotlib is optional; install pyfridge[plot] to enable plotting.
    """
    lambdas = np.concatenate([np.asarray(r["lambdas"], dtype=float) for r in cv.results_.values()])
    ax = _axes(ax)
    _lambda_scale(ax, lambdas, xscale)

    # Each candidate m can have its own grid. Plot actual evaluated points;
    # do not interpolate curves to infer additional CV candidates.
    for m, record in sorted(cv.results_.items()):
        grid = np.asarray(record["lambdas"])
        mean = np.asarray(record["mean_mse"])
        se = np.asarray(record["se_mse"])
        line, = ax.plot(grid, mean, linewidth=1.5, label=f"m={m}")
        if show_se:
            ax.fill_between(grid, mean - se, mean + se, color=line.get_color(), alpha=0.12)

    # Look up each choice at its own m and lambda; the two choices need not
    # share a curve. Put labels inside the axes, away from the markers, so
    # coincident selections remain readable and low-MSE labels cannot obscure
    # the lambda axis. Colors match each marker and its legend entry.
    choices = [("Minimum", cv.best_m_, cv.best_lambda_, "o", "#b2182b", (0.98, 0.96)),
               ("1-SE", cv.one_se_m_, cv.one_se_lambda_, "s", "#2166ac", (0.98, 0.88))]
    for label, m, lam, marker, color, offset in choices:
        record = cv.results_[m]
        matches = np.flatnonzero(np.asarray(record["lambdas"]) == lam)
        if matches.size != 1:
            raise ValueError(f"{label} lambda must occur exactly once in its CV grid")
        error = float(record["mean_mse"][matches[0]])
        ax.scatter([lam], [error], marker=marker, s=85, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=5, label=f"{label} (m={m})")
        ax.annotate(f"{label}: m={m}, lam={lam:.3g}", (lam, error),
                    xytext=offset, textcoords="axes fraction", color=color, fontsize=9,
                    ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    ax.axhline(cv.one_se_threshold_, linestyle="--", color="0.4", linewidth=1,
               label="Global 1-SE threshold")
    ax.set(xlabel=r"$\lambda$", ylabel="Mean CV MSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2),
              ncol=2, fontsize=9)
    return ax
