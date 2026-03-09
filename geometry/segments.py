import numpy as np


def point_to_segment_distance(px, py, ax, ay, bx, by):

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return float(np.hypot(px - ax, py - ay))

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, float(t)))

    cx = ax + t * abx
    cy = ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def segment_amplitude(xs, ys):

    # maximum distance from points on the segment to the chord connecting endpoints.

    ax, ay = float(xs[0]), float(ys[0])
    bx, by = float(xs[-1]), float(ys[-1])

    max_d = 0.0
    for px, py in zip(xs, ys):
        d = point_to_segment_distance(float(px), float(py), ax, ay, bx, by)
        if d > max_d:
            max_d = d
    return max_d


def build_cut_indices(n_points, infl_idx):

    # Build sorted cut indices including endpoints.
    # infl_idx are indices into x/y arrays.

    cut = np.concatenate(([0], np.asarray(infl_idx, dtype=int), [n_points - 1]))
    cut = np.unique(cut)
    cut.sort()
    return cut


def compute_segments(x, y, cut_idx):

    # Return a list of segment dicts between consecutive cut indices.
    # Each segment includes start/end indices and coordinate arrays.

    segments = []
    for i in range(len(cut_idx) - 1):
        a = int(cut_idx[i])
        b = int(cut_idx[i + 1])
        if b - a < 2:
            continue

        xs = x[a : b + 1]
        ys = y[a : b + 1]

        segments.append({
            "i0": a,
            "i1": b,
            "xs": xs,
            "ys": ys,
        })
    return segments


def classify_segments_by_amplitude(segments, wc, A_st_star=0.1):

    out = []
    n_arc = 0
    n_straight = 0

    for seg in segments:
        xs, ys = seg["xs"], seg["ys"]
        A_s = segment_amplitude(xs, ys)
        A_s_star = A_s / float(wc)

        is_arc = A_s_star > float(A_st_star)
        label = "arc" if is_arc else "straight"

        if is_arc:
            n_arc += 1
        else:
            n_straight += 1

        seg2 = dict(seg)
        seg2.update({
            "A_s": A_s,
            "A_s_star": A_s_star,
            "label": label
        })
        out.append(seg2)

    summary = {
        "n_segments": len(out),
        "n_arc": n_arc,
        "n_straight": n_straight,
        "arc_fraction": (n_arc / len(out)) if len(out) else 0.0,
    }
    return out, summary
