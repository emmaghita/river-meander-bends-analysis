import numpy as np
from scipy.signal import savgol_filter

def smooth_xy(x, y, window=21, poly=3):
    # Smooth x and y arrays with Savitzky–Golay filter.
    # window must be odd and < len(x).

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window % 2 == 0:
        raise ValueError("Savgol window length must be odd.")
    if len(x) <= window:
        raise ValueError(f"Not enough points ({len(x)}) for Savitzky–Golay window={window}")

    x_s = savgol_filter(x, window_length=window, polyorder=poly)
    y_s = savgol_filter(y, window_length=window, polyorder=poly)
    return x_s, y_s
