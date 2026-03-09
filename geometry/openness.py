import numpy as np

def _angle(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    c = np.dot(u, v) / (nu * nv)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def openness(x, y, i0, i1):
    if i1-i0 < 3:
        return np.nan

    p0 = np.array([x[i0], y[i0]])
    p1 = np.array([x[i1], y[i1]])

    t0 = np.array([x[i0+1] - x[i0], y[i0+1] - y[i0]])
    t1 = np.array([x[i1] - x[i1-1], y[i1] - y[i1-1]])

    chord = p1 - p0
    return max(_angle(t0, chord), _angle(t1, -chord))

def trim_to_openness(x, y, i0, i1, max_open=180):
    a, b = i0, i1
    while b - a > 4:
        op = openness(x, y, a, b)
        if np.isnan(op):
            a += 1
            b -= 1
            continue

        if op <= max_open:
            return a, b, op

        if _angle([x[a + 1] - x[a], y[a + 1] - y[a]], [x[b] - x[a], y[b] - y[a]]) >= \
                _angle([x[b] - x[b - 1], y[b] - y[b - 1]], [x[a] - x[b], y[a] - y[b]]):
            a += 1
        else:
            b -= 1

    return None, None, np.nan