import os
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np

from geometry import to_single_linestring


@dataclass
class LoadedDataset:
    source_path: str           # shp path actually used
    display_name: str          # for UI
    crs: str
    length: float              # units of CRS (meters only if projected!)
    vertex_count: int
    x: np.ndarray
    y: np.ndarray


class DatasetLoader:
    # Handles:
    # loading .shp directly
    # loading .zip containing a shapefile (extracts to a temp dir kept alive)
    def __init__(self):
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def resolve_to_shp(self, path: str) -> str:
        if path.lower().endswith(".shp"):
            return path
        if path.lower().endswith(".zip"):
            return self._extract_zip_find_shp(path)
        raise ValueError("Unsupported file type. Choose a .shp or a .zip shapefile.")

    def load_centerline(self, path: str) -> LoadedDataset:
        shp = self.resolve_to_shp(path)

        gdf = gpd.read_file(shp)
        if len(gdf) == 0:
            raise RuntimeError("Dataset contains no features.")

        geom = gdf.geometry.iloc[0]
        line = to_single_linestring(geom)

        coords = np.asarray(line.coords, dtype=float)
        x, y = coords[:, 0], coords[:, 1]

        crs = str(gdf.crs) if gdf.crs is not None else "Unknown"
        length = float(line.length)
        vertex_count = int(len(coords))

        return LoadedDataset(
            source_path=shp,
            display_name=os.path.basename(path),
            crs=crs,
            length=length,
            vertex_count=vertex_count,
            x=x,
            y=y
        )

    def _extract_zip_find_shp(self, zip_path: str) -> str:
        # keep extracted files alive while this loader exists
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="river_gui_")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._temp_dir.name)

        shp_files = []
        for root, _, files in os.walk(self._temp_dir.name):
            for fn in files:
                if fn.lower().endswith(".shp"):
                    shp_files.append(os.path.join(root, fn))

        if not shp_files:
            raise RuntimeError("No .shp found inside the ZIP.")
        return shp_files[0]
