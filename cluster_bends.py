# cluster_bends.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from clustering import (
    load_bends_table,
    save_bends_with_clusters,
    build_feature_matrix,
    pick_k_by_silhouette,
    fit_kmeans,
)


@dataclass
class ClusterRunResult:
    df: Any
    best_k: int
    scores: Dict[int, float]
    mask: Any
    labels: np.ndarray


def run_clustering(
    bends_csv: str = "outputs/bends_table.csv",
    *,
    auto_k: bool = True,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 0,
    n_init: int = 50,
    save_csv: bool = False,
    out_csv: str = "outputs/bends_with_clusters.csv",
) -> ClusterRunResult:
    """
    Run KMeans clustering on bends_table.csv and return a dataframe with a 'cluster' column.
    GUI-safe: no plt.show(), no plotting.
    """
    df = load_bends_table(bends_csv)

    X, mask, feature_names = build_feature_matrix(df, use_logAR=True, drop_nonfinite=True)

    if X.shape[0] == 0:
        raise ValueError("No finite rows available for clustering (all features non-finite).")

    if auto_k or k is None:
        best_k, scores = pick_k_by_silhouette(X, k_min=k_min, k_max=k_max, random_state=random_state)
        if best_k is None:
            raise ValueError("Could not select k by silhouette. Try fixed k.")
    else:
        best_k = int(k)
        scores = {}

    labels, model, scaler, Xs = fit_kmeans(X, k=best_k, random_state=random_state, n_init=n_init)

    df = df.copy()
    df["cluster"] = -1
    df.loc[mask, "cluster"] = labels

    if save_csv:
        save_bends_with_clusters(df, out_csv)

    return ClusterRunResult(df=df, best_k=best_k, scores=scores, mask=mask, labels=labels)


def make_cluster_map_figure(
    x_s: np.ndarray,
    y_s: np.ndarray,
    bends_final: List[Dict[str, Any]],
    clusters_df: Any,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    dpi: int = 150,
    draw_ticks: bool = True,
) -> plt.Figure:
    """
    Create the 'river + bends colored by cluster' map as a Matplotlib Figure.
    This matches the plot block you already had in main.py, but returns a Figure.
    """
    x_s = np.asarray(x_s, dtype=float)
    y_s = np.asarray(y_s, dtype=float)

    if x_s.shape != y_s.shape:
        raise ValueError("x_s and y_s must have the same shape.")

    if not {"i0", "i1", "cluster"}.issubset(set(clusters_df.columns)):
        raise ValueError("clusters_df must contain columns: i0, i1, cluster")

    cluster_map = {
        (int(r.i0), int(r.i1)): int(r.cluster)
        for r in clusters_df.itertuples(index=False)
    }

    bends = []
    for b in bends_final:
        b2 = dict(b)
        key = (int(b2["i0"]), int(b2["i1"]))
        b2["cluster"] = cluster_map.get(key, -1)
        bends.append(b2)

    clusters = sorted({int(b["cluster"]) for b in bends if int(b["cluster"]) >= 0})
    n_clusters = len(clusters)
    cmap = plt.get_cmap("tab10", max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x_s, y_s, linewidth=1.0, alpha=0.15)

    for b in bends:
        a, c = int(b["i0"]), int(b["i1"])
        cl = int(b["cluster"])

        a = max(0, min(a, len(x_s) - 1))
        c = max(0, min(c, len(x_s) - 1))
        if c <= a:
            continue

        if cl < 0:
            ax.plot(
                x_s[a:c + 1], y_s[a:c + 1],
                color="gray", linewidth=2.0, alpha=0.35
            )
        else:
            ci = clusters.index(cl)
            ax.plot(
                x_s[a:c + 1], y_s[a:c + 1],
                color=cmap(ci), linewidth=2.8, alpha=0.9
            )

    if draw_ticks:
        for b in bends:
            end_idx = int(b["i1"])
            draw_separator_tick(ax, x_s, y_s, end_idx)

    # Legend
    handles = [
        Line2D([0], [0], color=cmap(i), lw=3, label=f"Cluster {cl}")
        for i, cl in enumerate(clusters)
    ]
    handles.append(Line2D([0], [0], color="gray", lw=3, alpha=0.35, label="Unclustered"))

    ax.legend(handles=handles, framealpha=0.9)
    ax.set_title(title or "Final bends colored by KMeans cluster")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def run_cluster_map(
    x_s: np.ndarray,
    y_s: np.ndarray,
    bends_final: List[Dict[str, Any]],
    *,
    bends_csv: str = "outputs/bends_table.csv",
    auto_k: bool = True,
    k: Optional[int] = None,
    save_csv: bool = False,
    out_csv: str = "outputs/bends_with_clusters.csv",
) -> Tuple[plt.Figure, ClusterRunResult]:
    """
    Convenience: run clustering and immediately return the cluster-map Figure + run metadata.
    """
    res = run_clustering(
        bends_csv=bends_csv,
        auto_k=auto_k,
        k=k,
        save_csv=save_csv,
        out_csv=out_csv,
    )
    fig = make_cluster_map_figure(
        x_s, y_s, bends_final, res.df,
        title=f"Final bends colored by KMeans cluster (k={res.best_k})"
    )
    return fig, res


def draw_separator_tick(ax, x, y, idx, tick_len=60.0, lw=1.6):
    """
    Copied from your main.py behavior: draws a small perpendicular tick at a bend boundary.
    """
    if idx <= 0 or idx >= len(x) - 1:
        return

    x0, y0 = x[idx - 1], y[idx - 1]
    x1, y1 = x[idx + 1], y[idx + 1]
    tx, ty = (x1 - x0), (y1 - y0)

    n = float(np.hypot(tx, ty))
    if n == 0:
        return
    tx, ty = tx / n, ty / n

    # perpendicular normal
    nx, ny = -ty, tx
    cx, cy = x[idx], y[idx]
    h = tick_len / 2.0

    ax.plot(
        [cx - h * nx, cx + h * nx],
        [cy - h * ny, cy + h * ny],
        color="black",
        linewidth=lw,
        zorder=20
    )


if __name__ == "__main__":
    # Terminal-friendly: keep it, but don't require GUI objects.
    # This will still compute clusters (and optionally save CSV).
    res = run_clustering(save_csv=True)
    print(f"Loaded bends: {len(res.df)}")
    print(f"Chosen k: {res.best_k}")
    if res.scores:
        print("Silhouette scores:", {k: round(v, 4) for k, v in sorted(res.scores.items())})
    print("Saved: outputs/bends_with_clusters.csv")
