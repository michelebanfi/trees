"""Per-point cost-benefit studies for the shading proposal (no budget cap).

Shading footprints come from a hand-drawn GeoJSON (`data/<campus>/
placements.geojson`): one polygon/loop per proposal point, mapped to the
nearest point A/B/C. A and B are horizontal corridor rectangles; C is an
irregular N-S corridor loop (built as a series of rectangles).

For each point and each solution the footprint is filled to 25 / 50 / 100%
(coverage grows the structure along its long axis from the centre), with NO
budget cap:

  - velario: a translucent tensile canopy over the footprint at ~6 m;
  - trees  : semi-mature deciduous trees planted in TWO rows along the two long
             edges of the footprint, leaving a clear path down the middle.

Every scenario carries a receptor `window` (sized to the footprint), so
compute_sun_hours re-simulates only a small box around the point.

Run AFTER compute_sun_hours.py (baseline). Then re-simulate each JSON with
SCENARIO=<json> python3 compute_sun_hours.py, then run indicators.py.

Outputs: data/<campus>/scenario_pt{P}_{trees|sails}_{25|50|100}.json
"""
import json
import math
import os

import numpy as np
from matplotlib.path import Path

from campus_config import CFG, SUN_FILE, DATA_DIR
from compute_sun_hours import to_xy, RES
from scenarios import (Field, dilate, TREE_H, TREE_CROWN, TREE_SPACING,
                       TREE_BLDG_CLEAR, VELARIO_H, VELARIO_TAU)

GEOJSON = os.path.join(DATA_DIR, "placements.geojson")
WINDOW_MARGIN_M = 45.0            # receptor margin around the footprint (m)
LEVELS = [0.25, 0.50, 1.00]      # fraction of the structure length to build


def load_placements():
    """Read the drawn footprints; map each to the nearest proposal point."""
    gj = json.load(open(GEOJSON))
    polys = []
    for ft in gj["features"]:
        g = ft["geometry"]
        if g["type"] == "Polygon":
            ring = g["coordinates"][0]
        elif g["type"] == "LineString":
            ring = g["coordinates"]
        else:
            continue
        xy = np.array([to_xy(c[1], c[0]) for c in ring])   # GeoJSON = [lon,lat]
        polys.append(xy)

    pts = {n: to_xy(la, lo) for n, (la, lo) in CFG["proposal_points"].items()}
    out = {}
    for poly in polys:
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        name = min(pts, key=lambda n: (pts[n][0] - cx) ** 2
                   + (pts[n][1] - cy) ** 2)
        out[name] = poly
    return out


def long_axis(poly):
    """'x' (E-W corridor) or 'y' (N-S corridor) — the structure's long axis."""
    xspan = poly[:, 0].max() - poly[:, 0].min()
    yspan = poly[:, 1].max() - poly[:, 1].min()
    return "y" if yspan >= xspan else "x"


def footprint_mask(fld, poly, axis, level, ny, nx):
    """Cells inside the polygon and within the central `level` of its length."""
    xs, ys = poly[:, 0], poly[:, 1]
    ax = ys if axis == "y" else xs
    c = 0.5 * (ax.min() + ax.max())
    half = 0.5 * (ax.max() - ax.min()) * level
    amin, amax = c - half, c + half

    i0, j0 = fld.cell(xs.min(), ys.max())
    i1, j1 = fld.cell(xs.max(), ys.min())
    i0, i1 = max(0, i0), min(ny - 1, i1)
    j0, j1 = max(0, j0), min(nx - 1, j1)
    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
    X = fld.extent[0] + (jj + 0.5) * fld.res
    Y = fld.extent[3] - (ii + 0.5) * fld.res
    inside = Path(poly).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    axv = Y if axis == "y" else X
    inside &= (axv >= amin) & (axv <= amax)
    m = np.zeros((ny, nx), bool)
    m[ii[inside], jj[inside]] = True
    return m


def edge_row_trees(mask, fld, axis, plantable):
    """Two rows of trees along the footprint's long edges (central path free)."""
    step = max(1, int(round(TREE_SPACING / fld.res)))
    placed = []

    def keep(i, j):
        if not plantable[i, j]:
            return
        x, y = fld.xy(i, j)
        if any((x - px) ** 2 + (y - py) ** 2 < (0.8 * TREE_SPACING) ** 2
               for px, py in placed):
            return
        placed.append((x, y))

    if axis == "x":                          # long axis = columns; edges = rows
        cols = np.where(mask.any(axis=0))[0]
        for j in cols[::step]:
            rows = np.where(mask[:, j])[0]
            if len(rows):
                keep(rows.min(), j)
                keep(rows.max(), j)
    else:                                     # long axis = rows; edges = columns
        rows = np.where(mask.any(axis=1))[0]
        for i in rows[::step]:
            cols = np.where(mask[i])[0]
            if len(cols):
                keep(i, cols.min())
                keep(i, cols.max())
    return placed


def window_for(poly):
    """[centroid_lat, centroid_lon, half_x_m, half_y_m] covering the footprint."""
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    lat = CFG["origin"][0] + cy / 111132.0
    lon = CFG["origin"][1] + cx / (111320.0 * math.cos(math.radians(
        CFG["origin"][0])))
    half_x = 0.5 * (poly[:, 0].max() - poly[:, 0].min()) + WINDOW_MARGIN_M
    half_y = 0.5 * (poly[:, 1].max() - poly[:, 1].min()) + WINDOW_MARGIN_M
    return [round(lat, 6), round(lon, 6), round(half_x, 1), round(half_y, 1)]


def main():
    d = np.load(SUN_FILE, allow_pickle=False)
    fld = Field(d)
    bmask = d["bmask"]
    ny, nx = bmask.shape

    near_bldg = dilate(bmask, TREE_BLDG_CLEAR)
    occupied = np.zeros_like(bmask)
    for x, y, r in d["trees"]:
        i, j = fld.cell(x, y)
        if fld.inside(i, j):
            rr = max(1, int(round(r / RES)))
            g0, g1 = max(0, i - rr), min(ny, i + rr + 1)
            h0, h1 = max(0, j - rr), min(nx, j + rr + 1)
            occupied[g0:g1, h0:h1] = True
    plantable = ~bmask & ~near_bldg & ~occupied

    placements = load_placements()
    summary = []
    for name in CFG["proposal_points"]:
        if name not in placements:
            print(f"point {name}: no drawn footprint, skipped")
            continue
        poly = placements[name]
        axis = long_axis(poly)
        win = window_for(poly)
        full = footprint_mask(fld, poly, axis, 1.0, ny, nx)
        full_area = float((full & ~bmask).sum()) * RES * RES
        print(f"point {name}: {axis}-corridor, full footprint "
              f"{full_area:,.0f} m2, window half {win[2]:.0f}x{win[3]:.0f} m")

        for L in LEVELS:
            lvl = int(round(L * 100))
            mask = footprint_mask(fld, poly, axis, L, ny, nx) & ~bmask
            ci, cj = np.where(mask)
            cells_xy = [[round(float(x), 2), round(float(y), 2)]
                        for x, y in (fld.xy(i, j) for i, j in zip(ci, cj))]
            area = len(cells_xy) * RES * RES
            sc = {"name": f"pt{name}_sails_{lvl}", "solution": "sails",
                  "point": name, "level": L, "sail_cells": cells_xy,
                  "sail_h": VELARIO_H, "sail_tau": VELARIO_TAU,
                  "canopy_area_m2": round(area, 1), "window": win}
            json.dump(sc, open(f"{DATA_DIR}/scenario_pt{name}_sails_{lvl}.json",
                               "w"))

            trees = edge_row_trees(mask, fld, axis, plantable)
            tc = {"name": f"pt{name}_trees_{lvl}", "solution": "trees",
                  "point": name, "level": L,
                  "trees": [[round(tx, 2), round(ty, 2), TREE_H, TREE_CROWN]
                            for tx, ty in trees],
                  "n_trees": len(trees), "window": win}
            json.dump(tc, open(f"{DATA_DIR}/scenario_pt{name}_trees_{lvl}.json",
                               "w"))
            print(f"  {lvl:3d}% : velario {area:6,.0f} m2 | trees {len(trees):3d}")
            summary.append((name, lvl))

    print(f"\nwrote {len(summary) * 2} scenario files to {DATA_DIR}/")


if __name__ == "__main__":
    main()
