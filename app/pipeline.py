# app/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PipelineParams:
    wc: float = 40.0                       # characteristic width (m)
    spacing_m: Optional[float] = None      # default: wc

    smooth_window: int = 21
    smooth_poly: int = 3

    kappa_eps: float = 2e-4
    min_sep_m: Optional[float] = None      # default: 5 * wc

    A_st_star: float = 1.0                 # segment classification parameter

    max_search_arcs: int = 80              # Limaye mapping
    max_open: float = 120.0                # Limaye mapping
    A_min: float = 2.0                     # Limaye mapping


DEFAULT_BEND_COLS: List[str] = [
    "i0", "i1", "sign", "n_arcs", "is_compound",
    "C_m", "L_m", "S",
    "A_bend_m", "A_bend_star",
    "apex_idx", "A_up_m", "A_down_m", "AR",
    "openness", "baseline_m"
]

def _safe_stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(~np.isfinite(a)):
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.nanmin(a)),
        "median": float(np.nanmedian(a)),
        "max": float(np.nanmax(a)),
    }


def _count_true(vals: Sequence[bool]) -> int:
    return int(np.sum(np.asarray(vals, dtype=bool)))


def _lower(s: Any) -> str:
    return str(s).strip().lower()

def _get_geometry_api():
    try:
        from geometry.resample import resample_linestring
        from geometry.smoothing import smooth_xy
        from geometry.curvature import compute_curvature, threshold_curvature
        from geometry.inflections import (
            find_inflections,
            filter_inflections_by_separation,
        )
        from geometry.segments import (
            build_cut_indices,
            compute_segments,
            classify_segments_by_amplitude,
            segment_amplitude,
            point_to_segment_distance,
        )
        from geometry.bends import (
            build_bends_limaye,
            remove_contained_bends,
            summarize_bends,
            only_arcs,
        )
        from geometry.bend_metrics import compute_bend_metrics
        from geometry.merge import (
            add_segment_signs,
            merge_arc_straight_arc_same_sign,
        )

        return {
            "resample_linestring": resample_linestring,
            "smooth_xy": smooth_xy,
            "compute_curvature": compute_curvature,
            "threshold_curvature": threshold_curvature,
            "find_inflections": find_inflections,
            "filter_inflections_by_separation": filter_inflections_by_separation,
            "build_cut_indices": build_cut_indices,
            "compute_segments": compute_segments,
            "classify_segments_by_amplitude": classify_segments_by_amplitude,
            "segment_amplitude": segment_amplitude,
            "point_to_segment_distance": point_to_segment_distance,
            "build_bends_limaye": build_bends_limaye,
            "remove_contained_bends": remove_contained_bends,
            "summarize_bends": summarize_bends,
            "compute_bend_metrics": compute_bend_metrics,
            "add_segment_signs": add_segment_signs,
            "merge_arc_straight_arc_same_sign": merge_arc_straight_arc_same_sign,
            "only_arcs": only_arcs,
        }

    except Exception as e:
        raise ImportError(f"Failed to import geometry API: {e}")


# main pipeline

def run_bend_pipeline_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    params: PipelineParams = PipelineParams(),
) -> Dict[str, Any]:

    # Runs full pipeline starting from raw x,y polyline arrays (already in projected meters).
    g = _get_geometry_api()

    wc = float(params.wc)
    spacing_m = float(params.spacing_m if params.spacing_m is not None else wc)
    min_sep_m = float(params.min_sep_m if params.min_sep_m is not None else 5.0 * wc)

    try:
        # If i have shapely available and resampler expects a LineString:
        from shapely.geometry import LineString  # type: ignore
        line = LineString(np.column_stack([x, y]))
        resampled = g["resample_linestring"](line, spacing=spacing_m)
        coords = np.asarray(resampled.coords, dtype=float)
        x_r, y_r = coords[:, 0], coords[:, 1]
    except Exception:
        # fallback: assume resample_linestring can accept (x,y)
        x_r, y_r = g["resample_linestring"](x, y, spacing=spacing_m)  # type: ignore

    x_s, y_s = g["smooth_xy"](x_r, y_r, window=params.smooth_window, poly=params.smooth_poly)

    kappa = g["compute_curvature"](x_s, y_s, ds=spacing_m)
    kappa = g["threshold_curvature"](kappa, eps=params.kappa_eps)

    infl_raw = g["find_inflections"](kappa)
    infl = g["filter_inflections_by_separation"](infl_raw, ds=spacing_m, min_sep_m=min_sep_m)

    cut_idx = g["build_cut_indices"](len(x_s), infl)
    segments = g["compute_segments"](x_s, y_s, cut_idx)
    segments, _ = g["classify_segments_by_amplitude"](segments, wc=wc, A_st_star=params.A_st_star)
    segments = g["add_segment_signs"](segments, kappa)

    before_merge = len(segments)
    segments_merged = g["merge_arc_straight_arc_same_sign"](segments)
    after_merge = len(segments_merged)

    n_arc_m = sum(1 for s in segments_merged if str(s.get("label", "")).lower() == "arc")
    n_st_m = sum(1 for s in segments_merged if str(s.get("label", "")).lower() == "straight")
    seg_summary = {
        "n_segments": int(len(segments_merged)),
        "n_arc": int(n_arc_m),
        "n_straight": int(n_st_m),
        "arc_fraction": float(n_arc_m / len(segments_merged)) if len(segments_merged) else 0.0,
    }

    arcs = g["only_arcs"](segments_merged)

    bends_mapped = g["build_bends_limaye"](
        arcs, x_s, y_s,
        wc=wc,
        max_search_arcs=params.max_search_arcs,
        max_open=params.max_open,
        A_min=params.A_min,
    )
    bends_final = g["remove_contained_bends"](bends_mapped)
    bends_summary = g["summarize_bends"](bends_final)

    bend_table: List[Dict[str, Any]] = [g["compute_bend_metrics"](x_s, y_s, b, wc=wc) for b in bends_final]

    Astar = np.array([r.get("A_bend_star", np.nan) for r in bend_table], dtype=float)
    S = np.array([r.get("S", np.nan) for r in bend_table], dtype=float)
    AR = np.array([r.get("AR", np.nan) for r in bend_table], dtype=float)

    logAR = np.log(AR)
    logAR[~np.isfinite(logAR)] = np.nan


    n_arc = 0
    n_straight = 0
    for s in segments_merged:
        cls = _lower(s.get("cls", s.get("class", s.get("type", ""))))
        if "arc" in cls:
            n_arc += 1
        elif "straight" in cls:
            n_straight += 1

    n_segments = int(len(segments_merged))
    arc_fraction = float(n_arc / n_segments) if n_segments > 0 else float("nan")

    # bends simple/compound from bend_table if present, else from summary
    is_compound_list = [bool(r.get("is_compound", False)) for r in bend_table]
    n_compound = _count_true(is_compound_list)
    n_bends_final = int(len(bends_final))
    n_simple = n_bends_final - n_compound

    diagnostics = {
        "counts": {
            "spacing_m": float(spacing_m),
            "resampled_vertices": int(len(x_r)),
            "raw_inflections": int(len(infl_raw)),
            "filtered_inflections": int(len(infl)),
            "segments_before_merge": int(before_merge),
            "segments_after_merge": int(after_merge),
            "n_segments": n_segments,
            "n_arc": int(n_arc),
            "n_straight": int(n_straight),
            "arc_fraction": float(arc_fraction),
            "bends_mapped": int(len(bends_mapped)),
            "bends_final": n_bends_final,
            "n_simple": int(n_simple),
            "n_compound": int(n_compound),
        },
        "stats": {
            "A_bend_star": _safe_stats(Astar),
            "S": _safe_stats(S),
            "AR": _safe_stats(AR),
            "logAR": _safe_stats(logAR),
        }
    }

    params_used = asdict(params)
    params_used["spacing_m"] = spacing_m
    params_used["min_sep_m"] = min_sep_m

    return {
        "params_used": params_used,
        "series": {
            "x_raw": np.asarray(x, dtype=float),
            "y_raw": np.asarray(y, dtype=float),
            "x_resampled": np.asarray(x_r, dtype=float),
            "y_resampled": np.asarray(y_r, dtype=float),
            "x_s": np.asarray(x_s, dtype=float),
            "y_s": np.asarray(y_s, dtype=float),
            "kappa": np.asarray(kappa, dtype=float),
            "spacing_m": float(spacing_m),
        },
        "inflections": {
            "infl_raw": list(map(int, infl_raw)),
            "infl_filtered": list(map(int, infl)),
            "cut_idx": list(map(int, cut_idx)),
        },
        "segments": {
            "segments_merged": segments_merged,
            "summary": seg_summary,
        },
        "bends": {
            "bends_mapped": bends_mapped,
            "bends_final": bends_final,
            "summary": bends_summary,
        },
        "bend_table": bend_table,
        "diagnostics": diagnostics,
    }


def run_bend_pipeline_from_dataset(
    ds: Any,
    params: PipelineParams = PipelineParams(),
) -> Dict[str, Any]:
    x = np.asarray(ds.x, dtype=float)
    y = np.asarray(ds.y, dtype=float)
    return run_bend_pipeline_from_xy(x, y, params=params)

# Export helpers (use these from GUI buttons)

def export_bend_table_csv(
    bend_table: List[Dict[str, Any]],
    out_csv: str,
    cols: List[str] = DEFAULT_BEND_COLS,
) -> None:
    import os
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

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


def bend_table_to_rows(
    bend_table: List[Dict[str, Any]],
    cols: List[str] = DEFAULT_BEND_COLS,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for r in bend_table:
        rows.append([r.get(c, "") for c in cols])
    return rows
