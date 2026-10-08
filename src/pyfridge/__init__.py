"""Fridge regression and solution paths."""

# Re-export the supported public API so callers need not import implementation modules.
from ._coordinate import fit_fridge, FridgeResult
from ._path import path_fridge, FridgePath, refit_ols
from ._penalty import penalty, penalty_derivatives
from ._model import Fridge
from ._cv import cv_fridge, FridgeCV
from ._compat import (lm_fridge_cd, adafridge_solution_path,
                      differential_solution_path, gh_adaptive_fridge,
                      fridge_penalty_derivatives, final_ols, predict_fridgec)
from ._plot import plot_path_fridge, plot_cv_fridge


# Package metadata and the export list.
__version__ = "0.1.0"
__all__ = ["plot_path_fridge", "plot_cv_fridge", "Fridge", "fit_fridge", "path_fridge", "cv_fridge", "FridgeResult",
           "FridgePath", "FridgeCV", "penalty", "penalty_derivatives", "refit_ols",
           "lm_fridge_cd", "adafridge_solution_path", "differential_solution_path",
           "gh_adaptive_fridge", "fridge_penalty_derivatives", "final_ols", "predict_fridgec"]
