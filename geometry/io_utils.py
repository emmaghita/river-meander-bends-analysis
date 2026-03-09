from shapely.geometry import LineString
from shapely.ops import linemerge

def to_single_linestring(geometry):

    # Ensure geometry is a single LineString:
    # If MultiLineString: merge; if still multi, pick longest.

    if isinstance(geometry, LineString):
        return geometry

    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
        return max(list(merged), key=lambda g: g.length)

    raise TypeError(f"Expected LineString or MultiLineString but got {geometry.geom_type}")
