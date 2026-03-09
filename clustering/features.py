import numpy as np

def build_feature_matrix(df,
                         cols=("A_bend_star", "S", "AR"),
                         use_logAR=True,
                         drop_nonfinite=True):

    # ensure columns exist
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in bends table.")

    Astar = df[cols[0]].to_numpy(dtype=float)
    S = df[cols[1]].to_numpy(dtype=float)
    AR = df[cols[2]].to_numpy(dtype=float)

    if use_logAR:
        logAR = np.full_like(AR, np.nan, dtype=float)
        ok = np.isfinite(AR) & (AR > 0)
        logAR[ok] = np.log(AR[ok])
        X = np.column_stack([Astar, S, logAR])
        feature_names = ["A_bend_star", "S", "logAR"]
    else:
        X = np.column_stack([Astar, S, AR])
        feature_names = ["A_bend_star", "S", "AR"]

    if drop_nonfinite:
        mask = np.all(np.isfinite(X), axis=1)
        return X[mask], mask, feature_names

    mask = np.ones(len(df), dtype=bool)
    return X, mask, feature_names
