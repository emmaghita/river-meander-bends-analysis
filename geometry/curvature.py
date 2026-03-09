import numpy as np

def compute_curvature(x, y, ds):

    # Curvature kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    # using finite differences with respect to arclength s.
    # Assumes roughly constant spacing ds.

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = np.gradient(x, ds)
    dy = np.gradient(y, ds)
    ddx = np.gradient(dx, ds)
    ddy = np.gradient(dy, ds)

    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom == 0] = np.nan

    kappa = (dx * ddy - dy * ddx) / denom
    return kappa

def threshold_curvature(kappa, eps=2e-4):

    # Zero-out tiny curvature magnitudes to reduce micro sign flips.
    k = np.asarray(kappa, dtype=float).copy()
    k[np.abs(k) < eps] = 0.0
    return k
