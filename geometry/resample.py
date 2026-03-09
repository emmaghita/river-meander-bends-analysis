import numpy as np
from shapely.geometry import LineString

def resample_linestring(line: LineString, spacing: float) -> LineString:

    # Resample a LineString at roughly constant spacing using shapely.interpolate().
    # spacing must be in the line's CRS units (meters in EPSG:3844).

    length = line.length
    if length < spacing:
        return line

    distances = np.arange(0, length, spacing)
    distances = np.append(distances, length)
    distances = np.unique(distances)

    points = [line.interpolate(float(d)) for d in distances]
    return LineString(points)
