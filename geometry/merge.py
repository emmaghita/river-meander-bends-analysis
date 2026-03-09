import numpy as np

def segment_sign_from_kappa(i0, i1, kappa, eps=0.0):

    # Sign of a segment based on mean curvature.
    # Returns +1, -1, or 0.

    vals = np.asarray(kappa[i0:i1+1], dtype=float)
    if eps > 0:
        vals = vals.copy()
        vals[np.abs(vals) < eps] = 0.0
    m = np.nanmean(vals)
    if m > 0:
        return 1
    if m < 0:
        return -1
    return 0

def add_segment_signs(segments, kappa):
    # arc segments will have sign +/-1, straights will be 0.
    out = []
    for seg in segments:
        seg2 = dict(seg)
        if seg2.get("label") == "arc":
            seg2["sign"] = segment_sign_from_kappa(seg2["i0"], seg2["i1"], kappa)
        else:
            seg2["sign"] = 0
        out.append(seg2)
    return out

def merge_arc_straight_arc_same_sign(segments):

    # Merge pattern: arc(s) - straight - arc(s) where both arcs have same sign.

    merged = []
    i = 0
    while i < len(segments):
        # need at least 3 segments to match pattern
        if i + 2 < len(segments):
            s1, s2, s3 = segments[i], segments[i+1], segments[i+2]

            cond = (
                s1["label"] == "arc" and
                s2["label"] == "straight" and
                s3["label"] == "arc" and
                s1.get("sign", 0) != 0 and
                s1.get("sign", 0) == s3.get("sign", 0)
            )

            if cond:
                # Merge into one arc spanning i0..i1 across all three
                new_seg = {
                    "i0": s1["i0"],
                    "i1": s3["i1"],
                    "label": "arc",
                    "sign": s1["sign"],
                    # keep debug info
                    "merged_from": (i, i+1, i+2),
                }
                merged.append(new_seg)
                i += 3
                continue

        # default: keep segment
        merged.append(segments[i])
        i += 1

    return merged
