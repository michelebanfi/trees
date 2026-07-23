"""Walkable / activity polygons for the campus (areas.geojson).

A colleague mapped the parts of the campus that are actually walkable and where
people spend time. These helpers parse those polygons, rasterise them onto the
sun-hours grid (union mask + per-polygon masks) and locate the campus centre,
so indicators.py can:
  - report the campus context restricted to this reduced, more relevant area;
  - rank each polygon by sun exposure / area / centrality to point future work
    at the areas that matter most.

Only Polygon features are used; stray Point features in the file are ignored.
"""
import json
import os

import numpy as np
from matplotlib.path import Path

from campus_config import DATA_DIR
from compute_sun_hours import to_xy

AREAS_FILE = os.path.join(DATA_DIR, "areas.geojson")


def _shoelace(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_areas():
    """Return the activity polygons as [{xy (local m), area (m2)}...]."""
    gj = json.load(open(AREAS_FILE))
    out = []
    for ft in gj["features"]:
        g = ft["geometry"]
        if g["type"] != "Polygon":
            continue
        xy = np.array([to_xy(c[1], c[0]) for c in g["coordinates"][0]])
        out.append({"xy": xy, "area": float(_shoelace(xy))})
    return out


def poly_mask(fld, xy, ny, nx):
    """Boolean grid mask of the cells whose centre falls inside a polygon."""
    xs, ys = xy[:, 0], xy[:, 1]
    i0, j0 = fld.cell(xs.min(), ys.max())
    i1, j1 = fld.cell(xs.max(), ys.min())
    i0, i1 = max(0, i0), min(ny - 1, i1)
    j0, j1 = max(0, j0), min(nx - 1, j1)
    m = np.zeros((ny, nx), bool)
    if i0 > i1 or j0 > j1:
        return m
    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
    X = fld.extent[0] + (jj + 0.5) * fld.res
    Y = fld.extent[3] - (ii + 0.5) * fld.res
    inside = Path(xy).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    m[ii[inside], jj[inside]] = True
    return m


def build_masks(fld, bmask):
    """Per-polygon and union masks (ground only) + the campus activity centre.

    Returns (areas, union_ground, center_xy) where each area dict gains
    'mask' (ground cells) and 'cxy' (its ground centroid, local m)."""
    ny, nx = bmask.shape
    areas = load_areas()
    union = np.zeros((ny, nx), bool)
    for a in areas:
        m = poly_mask(fld, a["xy"], ny, nx) & ~bmask
        a["mask"] = m
        union |= m
        ii, jj = np.where(m if m.any() else poly_mask(fld, a["xy"], ny, nx))
        xy = np.array([fld.xy(i, j) for i, j in zip(ii, jj)])
        a["cxy"] = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))
    ui, uj = np.where(union)
    uxy = np.array([fld.xy(i, j) for i, j in zip(ui, uj)])
    center = (float(uxy[:, 0].mean()), float(uxy[:, 1].mean()))
    return areas, union, center
