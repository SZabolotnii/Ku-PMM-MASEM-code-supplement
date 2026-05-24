"""
PMM-MASEM: Variance-reduced manifold sampling via polynomial-maximization
density estimation.

Package structure:
  masem.estimators  — unified density estimator interface (Plugin, k_Ensemble,
                      MLE-Exp, PMM2/MLE)
  masem.spacings    — kNN shell-spacing utilities
  masem.resampling  — MASEM resampling step
"""

from masem.estimators import (
    DensityEstimator,
    ESTIMATOR_REGISTRY,
    PluginEstimator,
    kEnsembleEstimator,
    MLEExpEstimator,
    PMMEstimator,
    get_estimator,
    _weights_from_density,
)
from masem.spacings import (
    unit_ball_volume,
    knn_distances,
    shell_spacings,
    knn_spacings,
)

__all__ = [
    # Estimators
    "DensityEstimator",
    "ESTIMATOR_REGISTRY",
    "PluginEstimator",
    "kEnsembleEstimator",
    "MLEExpEstimator",
    "PMMEstimator",
    "get_estimator",
    "_weights_from_density",
    # Spacings
    "unit_ball_volume",
    "knn_distances",
    "shell_spacings",
    "knn_spacings",
]
