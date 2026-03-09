import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


from geometry import (
    to_single_linestring,
    resample_linestring,
    smooth_xy,
    compute_curvature,
    threshold_curvature,
    find_inflections,
    filter_inflections_by_separation,
    build_cut_indices,
    compute_segments,
    classify_segments_by_amplitude,
    add_segment_signs,
    merge_arc_straight_arc_same_sign,
    only_arcs,
    build_bends_limaye,
    remove_contained_bends,
    summarize_bends,
    compute_bend_metrics,   # <-- NEW
)

def summarize_segments(segs):
    n = len(segs)
    n_arc = sum(1 for s in segs if s.get("label") == "arc")
    n_straight = sum(1 for s in segs if s.get("label") == "straight")
    return {
        "n_segments": n,
        "n_arc": n_arc,
        "n_straight": n_straight,
        "arc_fraction": (n_arc / n) if n else 0.0
    }

def main():
    gdf = gpd.read_file("data/somesul_mic.shp")
    print("Original CRS:", gdf.crs)

    line = to_single_linestring(gdf.geometry.iloc[0])
    print(f"Original length (m): {line.length:.1f}")
    print(f"Original vertex count: {len(line.coords)}")

    wc = 40.0
    spacing_m = wc

    smooth_window = 21
    smooth_poly = 3

    kappa_eps = 2e-4
    min_sep_m = 5 * wc

    A_st_star = 1.0
    max_search_arcs = 80
    max_open = 120
    A_min = 2.0

    resampled = resample_linestring(line, spacing=spacing_m)
    coords = np.asarray(resampled.coords)
    x, y = coords[:, 0], coords[:, 1]

    print(f"Resampling spacing (m): {spacing_m}")
    print(f"Resampled vertex count: {len(x)}")

    x_s, y_s = smooth_xy(x, y, window=smooth_window, poly=smooth_poly)

    kappa = compute_curvature(x_s, y_s, ds=spacing_m)
    kappa = threshold_curvature(kappa, eps=kappa_eps)

    infl_raw = find_inflections(kappa)
    infl = filter_inflections_by_separation(infl_raw, ds=spacing_m, min_sep_m=min_sep_m)

    print(f"Raw inflections: {len(infl_raw)}")
    print(f"Filtered inflections (min_sep={min_sep_m:.0f} m): {len(infl)}")

    cut_idx = build_cut_indices(len(x_s), infl)
    segments = compute_segments(x_s, y_s, cut_idx)
    segments, _ = classify_segments_by_amplitude(segments, wc=wc, A_st_star=A_st_star)

    segments = add_segment_signs(segments, kappa)

    before = len(segments)
    segments_merged = merge_arc_straight_arc_same_sign(segments)
    after = len(segments_merged)

    print(f"\nMerge arc–straight–arc (same sign): {before} -> {after}")

    print("\nSegment classification summary (AFTER merge):")
    for k, v in summarize_segments(segments_merged).items():
        print(f"  {k}: {v}")

    arcs = only_arcs(segments_merged)

    bends_mapped = build_bends_limaye(
        arcs, x_s, y_s,
        wc=wc,
        max_search_arcs=max_search_arcs,
        max_open=max_open,
        A_min=A_min
    )

    bends_final = remove_contained_bends(bends_mapped)

    print(f"\nMapped bends (Limaye): {len(bends_mapped)}")
    print(f"After remove-contained: {len(bends_final)}")

    print("\nBends summary (FINAL, Limaye):")
    for k, v in summarize_bends(bends_final).items():
        print(f"  {k}: {v}")

    bend_table = [compute_bend_metrics(x_s, y_s, b, wc=wc) for b in bends_final]

    Astar = np.array([r["A_bend_star"] for r in bend_table], dtype=float)
    S = np.array([r["S"] for r in bend_table], dtype=float)
    AR = np.array([r["AR"] for r in bend_table], dtype=float)

    logAR = np.log(AR)

    logAR[~np.isfinite(logAR)] = np.nan

    print("\nFINAL bend metrics diagnostics:")
    print(f"  A_bend*: min={np.nanmin(Astar):.2f}, median={np.nanmedian(Astar):.2f}, max={np.nanmax(Astar):.2f}")
    print(f"  Sinuosity: min={np.nanmin(S):.3f}, median={np.nanmedian(S):.3f}, max={np.nanmax(S):.3f}")
    print(f"  AR: min={np.nanmin(AR):.3f}, median={np.nanmedian(AR):.3f}, max={np.nanmax(AR):.3f}")

    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/bends_table.csv"

    cols = [
        "i0", "i1", "sign", "n_arcs", "is_compound",
        "C_m", "L_m", "S",
        "A_bend_m", "A_bend_star",
        "apex_idx", "A_up_m", "A_down_m", "AR",
        "openness", "baseline_m"
    ]

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in bend_table:
            row = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, bool):
                    v = int(v)
                row.append(str(v))
            f.write(",".join(row) + "\n")

    print(f"\nSaved bend table to: {out_csv}")

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.plot(x_s, y_s, linewidth=1.0, alpha=0.15)

    for b in bends_final:
        a, c = b["i0"], b["i1"]
        color = "tab:blue" if b["sign"] > 0 else "tab:orange"
        ax.plot(x_s[a:c + 1], y_s[a:c + 1], linewidth=2.8, color=color)

    ax.set_title("Final bends (Limaye-mapped)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.5), dpi=150)
    ax2.hist(Astar[np.isfinite(Astar)], bins=30)
    ax2.set_title("Histogram: A_bend* (final bends)")
    ax2.set_xlabel("A_bend*")
    ax2.set_ylabel("count")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(6.5, 3.5), dpi=150)
    ax3.scatter(Astar, S, s=18, alpha=0.8)
    ax3.set_title("A_bend* vs Sinuosity (final bends)")
    ax3.set_xlabel("A_bend*")
    ax3.set_ylabel("S")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(6.5, 3.5), dpi=150)
    ax4.scatter(Astar, logAR, s=18, alpha=0.8)
    ax4.set_title("A_bend* vs log(AR) (final bends)")
    ax4.set_xlabel("A_bend*")
    ax4.set_ylabel("log(AR)")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    cluster_csv = "outputs/bends_with_clusters.csv"

    if not os.path.exists(cluster_csv):
        print("\n[Cluster plot skipped] Run cluster_bends.py first.")
    else:
        dfc = pd.read_csv(cluster_csv)

        # map (i0, i1) -> cluster
        cluster_map = {
            (int(r.i0), int(r.i1)): int(r.cluster)
            for r in dfc.itertuples()
        }

        # attach cluster to bends
        for b in bends_final:
            key = (int(b["i0"]), int(b["i1"]))
            b["cluster"] = cluster_map.get(key, -1)

        # unique clusters (ignore -1)
        clusters = sorted({b["cluster"] for b in bends_final if b["cluster"] >= 0})
        n_clusters = len(clusters)

        cmap = plt.get_cmap("tab10", max(n_clusters, 1))

        figc, axc = plt.subplots(figsize=(14, 7), dpi=150)
        axc.plot(x_s, y_s, linewidth=1.0, alpha=0.15)

        for b in bends_final:
            a, c = int(b["i0"]), int(b["i1"])
            cl = b["cluster"]

            if cl < 0:
                axc.plot(
                    x_s[a:c + 1], y_s[a:c + 1],
                    color="gray", linewidth=2.0, alpha=0.35
                )
            else:
                ci = clusters.index(cl)
                axc.plot(
                    x_s[a:c + 1], y_s[a:c + 1],
                    color=cmap(ci), linewidth=2.8, alpha=0.9
                )
        for b in bends_final:
            end_idx = int(b["i1"])
            draw_separator_tick(axc, x_s, y_s, end_idx)

        # legend
        handles = [
            Line2D([0], [0], color=cmap(i), lw=3, label=f"Cluster {cl}")
            for i, cl in enumerate(clusters)
        ]
        handles.append(Line2D([0], [0], color="gray", lw=3, alpha=0.35,
                              label="Unclustered"))

        axc.legend(handles=handles, framealpha=0.9)
        axc.set_title("Final bends colored by KMeans cluster")
        axc.set_xlabel("X (m)")
        axc.set_ylabel("Y (m)")
        axc.axis("equal")
        axc.grid(True, alpha=0.3)

        plt.tight_layout()

        out_pdf = "outputs/bends_clusters_map.pdf"
        plt.savefig(out_pdf)
        print(f"\nSaved cluster map to: {out_pdf}")

        plt.show()

def draw_separator_tick(ax, xs, ys, idx, tick_len=60, lw=1.6):
    if idx <= 0 or idx >= len(xs) - 1:
        return

    x0, y0 = xs[idx - 1], ys[idx - 1]
    x1, y1 = xs[idx + 1], ys[idx + 1]
    tx, ty = (x1 - x0), (y1 - y0)

    n = np.hypot(tx, ty)
    if n == 0:
        return
    tx, ty = tx / n, ty / n

    nx, ny = -ty, tx

    cx, cy = xs[idx], ys[idx]
    h = tick_len / 2.0

    ax.plot(
        [cx - h * nx, cx + h * nx],
        [cy - h * ny, cy + h * ny],
        color="black",
        linewidth=lw,
        zorder=20
    )


if __name__ == "__main__":
    main()

