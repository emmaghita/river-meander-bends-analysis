import numpy as np

def _point_segment_distance(px, py, ax, ay, bx, by):
    # distance from P to seg AB
    abx, aby = bx - ax, by-ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return np.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy)

def chord_length(x, y, i0, i1):
    return float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]))

def arc_length(x, y, i0, i1):
    xs = x[i0:i1+1]
    ys = y[i0:i1+1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))

def distances_to_chord(x, y, i0, i1):
    ax, ay = x[i0], y[i0]
    bx, by = x[i1], y[i1]
    idx = np.arange(i0, i1 + 1)
    d = np.array([_point_segment_distance(x[k], y[k], ax, ay, bx, by) for k in idx], dtype=float)
    return idx, d

def bend_amplitude_and_apex(x, y, i0, i1):
    idx, d = distances_to_chord(x, y, i0, i1)
    if len(d) == 0:
        return np.nan, None
    k = int(np.argmax(d))
    apex_idx = int(idx[k])
    A = float(d[k])
    return A, apex_idx

def bend_asymmetry(x, y, i0, i1, apex_idx):
    idx, d = distances_to_chord(x, y, i0, i1)
    if apex_idx is None:
        return np.nan, np.nan, np.nan

    mask_up = idx < apex_idx
    mask_down = idx > apex_idx

    if not np.any(mask_up) or not np.any(mask_down):
        return np.nan, np.nan, np.nan

    A_up = float(np.max(d[mask_up]))
    A_down = float(np.max(d[mask_down]))

    if not np.isfinite(A_up) or not np.isfinite(A_down) or A_down == 0:
        return np.nan, np.nan, np.nan

    AR = A_up / A_down
    return A_up, A_down, AR

def compute_bend_metrics(x, y, bend, wc):
    i0, i1 = int(bend["i0"]), int(bend["i1"])
    C = chord_length(x, y, i0, i1)
    L = arc_length(x, y, i0, i1)
    S = (L / C) if C>0 else np.nan

    A, apex_idx = bend_amplitude_and_apex(x, y, i0, i1)
    A_star = (A / float(wc)) if np.isfinite(A) else np.nan

    A_up, A_down, AR = bend_asymmetry(x, y, i0, i1, apex_idx)

    out = dict(bend)
    out.update({
        "C_m": C,
        "L_m": L,
        "S": S,
        "apex_idx": apex_idx,
        "A_bend_m": A,
        "A_bend_star": A_star,
        "A_up_m": A_up,
        "A_down_m": A_down,
        "AR": AR
    })
    return out