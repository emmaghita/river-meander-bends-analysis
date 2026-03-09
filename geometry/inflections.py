import numpy as np

def find_inflections(kappa):
    k = np.asarray(kappa, dtype=float).copy()
    sgn = np.sign(k)

    # fill zeros with previous sign to avoid fake crossings
    for i in range(1, len(sgn)):
        if sgn[i] == 0:
            sgn[i] = sgn[i - 1]

    infl_idx = np.where(sgn[:-1] * sgn[1:] < 0)[0] + 1
    return infl_idx

def filter_inflections_by_separation(infl_idx, ds, min_sep_m):
    infl_idx = np.asarray(infl_idx, dtype=int)
    if len(infl_idx) == 0:
        return infl_idx

    keep = [infl_idx[0]]
    last = infl_idx[0]
    for idx in infl_idx[1:]:
        if (idx - last) * ds >= min_sep_m:
            keep.append(idx)
            last = idx
    return np.array(keep, dtype=int)
