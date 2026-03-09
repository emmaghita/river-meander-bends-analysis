from .openness import trim_to_openness
from .segments import segment_amplitude
import numpy as np

def only_arcs(segments):

    # Keep only segments that are arcs and have a non-zero curvature sign.
    return [s for s in segments if s.get("label") == "arc" and s.get("sign", 0) != 0]


def build_candidate_bends(arcs, max_search_arcs=80):

    # Build candidate bends from an arc sequence.

    bends = []
    n = len(arcs)

    for i in range(n):
        s0 = arcs[i]["sign"]
        seen_opposite = False

        for j in range(i + 1, min(n, i + 1 + max_search_arcs)):
            sj = arcs[j]["sign"]

            if sj == -s0:
                seen_opposite = True
                continue

            if sj == s0 and seen_opposite:
                n_arcs = j - i + 1
                bends.append({
                    "arc_i0": i,
                    "arc_i1": j,
                    "i0": arcs[i]["i0"],
                    "i1": arcs[j]["i1"],
                    "sign": s0,
                    "n_arcs": n_arcs,
                    "is_compound": n_arcs > 3
                })
                break  # stop at the first closing arc

    return bends

def filter_bends(bends, x, y, wc, A_min=1.0, max_open=180):
    out = []
    for b in bends:
        i0, i1 = b["i0"], b["i1"]

        a, c, op = trim_to_openness(x, y, i0, i1, max_open=max_open)
        if a is None:
            continue

        xs = x[a:c + 1]
        ys = y[a:c + 1]

        A_bend = segment_amplitude(xs, ys)
        A_star = A_bend / float(wc)

        if A_star < float(A_min):
            continue

        b2 = dict(b)
        b2["i0"] = a
        b2["i1"] = c
        b2["openness"] = op
        b2["A_bend_m"] = A_bend
        b2["A_bend_star"] = A_star
        out.append(b2)

    return out


def summarize_bends(bends):
    return {
        "n_bends": len(bends),
        "n_simple": sum(1 for b in bends if not b.get("is_compound", False)),
        "n_compound": sum(1 for b in bends if b.get("is_compound", False)),
    }

def _baseline_len(x, y, i0, i1):
    return float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]))


def build_bends_limaye(arcs, x, y, wc, max_search_arcs=80, max_open=120, A_min=1.0):

    # Returns a list of bends (already trimmed + strength-filtered).

    bends = []
    n = len(arcs)

    # ensure numpy arrays for fast indexing
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    for i in range(n):
        s0 = arcs[i]["sign"]
        seen_opposite = False

        best = None
        best_baseline = np.inf

        # start/end indices in the centerline point array
        start_i0 = arcs[i]["i0"]

        for j in range(i + 1, min(n, i + 1 + max_search_arcs)):
            sj = arcs[j]["sign"]

            if sj == -s0:
                seen_opposite = True
                continue

            if sj == s0 and seen_opposite:
                end_i1 = arcs[j]["i1"]

                # 1) openness trimming (your openness proxy)
                a, c, op = trim_to_openness(x, y, start_i0, end_i1, max_open=max_open)
                if a is None:
                    continue

                # 2) bend strength on trimmed window
                xs = x[a:c + 1]
                ys = y[a:c + 1]
                A_bend = segment_amplitude(xs, ys)
                A_star = A_bend / float(wc)
                if A_star < float(A_min):
                    continue

                # 3) baseline length (Limaye selection key)
                bl = _baseline_len(x, y, a, c)

                if bl < best_baseline:
                    n_arcs = j - i + 1
                    best_baseline = bl
                    best = {
                        "arc_i0": i,
                        "arc_i1": j,
                        "i0": a,
                        "i1": c,
                        "sign": s0,
                        "n_arcs": n_arcs,
                        "is_compound": n_arcs > 3,
                        "openness": float(op),
                        "A_bend_m": float(A_bend),
                        "A_bend_star": float(A_star),
                        "baseline_m": float(bl),
                    }

        if best is not None:
            bends.append(best)

    return bends


def remove_contained_bends(bends):

    # Remove bends fully contained within other bends
    # Keep larger containers, drop nested ones
    if not bends:
        return bends

    # sort by start increasing, then length decreasing (so containers come first)
    bends_sorted = sorted(bends, key=lambda b: (b["i0"], -(b["i1"] - b["i0"])))

    keep = []
    for b in bends_sorted:
        contained = False
        for k in keep:
            if k["i0"] <= b["i0"] and b["i1"] <= k["i1"]:
                contained = True
                break
        if not contained:
            keep.append(b)

    # return in downstream order
    return sorted(keep, key=lambda b: b["i0"])
