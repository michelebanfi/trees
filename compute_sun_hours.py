"""Compute annual direct-sun exposure for PoliMi Citta Studi.

Pipeline: OSM footprints + Milano DBT elevations + OSM/municipality/DBT trees
-> 2.5D obstacle rasters -> solar positions across the year -> per-cell shadow
ray-marching with canopy transmissivity -> sun hours, clear-sky insolation and
lunch-window (12-14 local) exposure, monthly.

Model (v2):
- Flat terrain (Milan). Receptors at ground level.
- Building footprints from OSM (correct courtyard topology); heights from the
  DBT 3D survey: 75th percentile of roof elevations inside the footprint minus
  local ground elevation. DBT volumes with no OSM footprint are added as extra
  prisms. OSM tags/type defaults only as fallback.
- Tree crowns are ellipsoids (crown base to top), semi-transparent:
  direct transmissivity 0.15 in leaf, 0.55 bare (deciduous Nov-Mar).
  Positions: OSM + municipality census + DBT survey (deduplicated at 8 m);
  sizes from municipality attributes (nearest <= 8 m), else local medians.
- Irradiance: clear-sky DNI = 1361 * 0.7^(AM^0.678), Kasten-Young air mass;
  horizontal direct = DNI * sin(elev). No diffuse component.
"""
import json
import math
import numpy as np

from campus_config import (CFG, OSM_FILE, COVER_FILE, MUNI_FILE, DBT_FILE,
                           SUN_FILE as OUT_FILE)

# --- geometry ------------------------------------------------------------
LAT0, LON0 = CFG["origin"]                       # projection origin
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))

RES = 1.5                                        # raster resolution (m)
# receptor (analysis) box
REC_S, REC_W, REC_N, REC_E = CFG["analysis_bbox"]

KCAP = 220                                       # max ray steps (=330 m reach)
MIN_ELEV = 3.0                                   # deg
STEP_MIN = 15                                    # time step (minutes)
SAMPLE_DAY = 15
LEAF_OFF_MONTHS = {11, 12, 1, 2, 3}

TAU_LEAF = 0.15    # direct transmissivity of a leafed crown
TAU_BARE = 0.55    # bare deciduous crown (branches only)
SOLAR_CONST = 1361.0

POIS = CFG["pois"]
POI_RADIUS = 15.0

BUILDING_DEFAULT_H = {
    "apartments": 19.0, "residential": 16.0, "university": 15.0,
    "hospital": 18.0, "church": 15.0, "school": 12.0, "office": 15.0,
    "commercial": 12.0, "retail": 8.0, "industrial": 8.0,
    "garages": 3.0, "garage": 3.0, "roof": 4.0, "shed": 3.0,
    "service": 4.0, "kiosk": 3.0, "hut": 3.0, "carport": 3.0,
}
LEVEL_H = 3.2
LEVEL_H_INSTITUTIONAL = 4.2
INSTITUTIONAL = {"university", "hospital", "church", "school", "public"}

# ground cover classes and render/heat metadata
COVER = {"paved": 0, "asphalt": 1, "footpath": 2, "grass": 3, "water": 4,
         "sport": 5, "building": 9}
ROAD_WIDTH = {"motorway": 16, "trunk": 14, "primary": 12, "secondary": 10,
              "tertiary": 8, "unclassified": 6, "residential": 6,
              "living_street": 5, "service": 4, "track": 3,
              "pedestrian": 5, "footway": 2.5, "cycleway": 2.5,
              "path": 2, "steps": 2}
PEDESTRIAN = {"pedestrian", "footway", "path", "steps"}


def to_xy(lat, lon):
    return (lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT


def parse_num(s):
    if s is None:
        return None
    try:
        return float(str(s).replace("m", "").replace(",", ".").strip())
    except ValueError:
        return None


# --- UTM 32N (DBT) -> local xy --------------------------------------------
def _utm32_forward(lat, lon):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); k0 = 0.9996
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians(9.0)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = ep2 * math.cos(phi) ** 2
    A = (lam - lam0) * math.cos(phi)
    m = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*phi)
             + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*phi)
             - (35*e2**3/3072) * math.sin(6*phi))
    E = k0 * n * (A + (1 - t + c) * A**3/6
                  + (5 - 18*t + t**2 + 72*c - 58*ep2) * A**5/120) + 500000.0
    N = k0 * (m + n * math.tan(phi) * (A**2/2 + (5 - t + 9*c + 4*c**2) * A**4/24
              + (61 - 58*t + t**2 + 600*c - 330*ep2) * A**6/720))
    return E, N


def make_utm_to_xy():
    """Affine fit of UTM32N->local over the study area (residuals < 1 cm)."""
    s, w, n, e = CFG["analysis_bbox"]
    src, dst = [], []
    for lat in (s, (s + n) / 2, n):
        for lon in (w, (w + e) / 2, e):
            src.append(_utm32_forward(lat, lon))
            dst.append(to_xy(lat, lon))
    src = np.array(src); dst = np.array(dst)
    A = np.column_stack([src, np.ones(len(src))])
    cx, rx, *_ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
    cy, ry, *_ = np.linalg.lstsq(A, dst[:, 1], rcond=None)

    def conv(E, N):
        E = np.asarray(E, float); N = np.asarray(N, float)
        return cx[0]*E + cx[1]*N + cx[2], cy[0]*E + cy[1]*N + cy[2]
    return conv


# --- solar position (NOAA, vectorised) ------------------------------------
def solar_position(jd_frac, lat, lon):
    jc = (jd_frac - 2451545.0) / 36525.0
    gmls = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    rad, deg = np.radians, np.degrees
    eoc = (np.sin(rad(gmas)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
           + np.sin(rad(2 * gmas)) * (0.019993 - 0.000101 * jc)
           + np.sin(rad(3 * gmas)) * 0.000289)
    stl = gmls + eoc
    sal = stl - 0.00569 - 0.00478 * np.sin(rad(125.04 - 1934.136 * jc))
    moe = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    oc = moe + 0.00256 * np.cos(rad(125.04 - 1934.136 * jc))
    decl = deg(np.arcsin(np.sin(rad(oc)) * np.sin(rad(sal))))
    vary = np.tan(rad(oc / 2)) ** 2
    eqtime = 4 * deg(vary * np.sin(2 * rad(gmls)) - 2 * eeo * np.sin(rad(gmas))
                     + 4 * eeo * vary * np.sin(rad(gmas)) * np.cos(2 * rad(gmls))
                     - 0.5 * vary ** 2 * np.sin(4 * rad(gmls))
                     - 1.25 * eeo ** 2 * np.sin(2 * rad(gmas)))
    minutes = (jd_frac + 0.5 - np.floor(jd_frac + 0.5)) * 1440
    tst = (minutes + eqtime + 4 * lon) % 1440
    ha = np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180)
    zen = deg(np.arccos(np.clip(
        np.sin(rad(lat)) * np.sin(rad(decl))
        + np.cos(rad(lat)) * np.cos(rad(decl)) * np.cos(rad(ha)), -1, 1)))
    elev = 90 - zen
    az_arg = np.clip((np.sin(rad(lat)) * np.cos(rad(zen)) - np.sin(rad(decl)))
                     / (np.cos(rad(lat)) * np.sin(rad(zen))), -1, 1)
    az = np.where(ha > 0, (deg(np.arccos(az_arg)) + 180) % 360,
                  (540 - deg(np.arccos(az_arg))) % 360)
    return elev, az


def julian_date(year, month, day, minutes_utc):
    import datetime
    d = datetime.date(year, month, day)
    return d.toordinal() + 1721424.5 + minutes_utc / 1440.0


def clear_sky_direct_horizontal(elev_deg):
    """Clear-sky direct irradiance on a horizontal plane, W/m^2."""
    el = max(elev_deg, 0.0)
    am = 1.0 / (math.sin(math.radians(el))
                + 0.50572 * (el + 6.07995) ** -1.6364)
    dni = SOLAR_CONST * 0.7 ** (am ** 0.678)
    return dni * math.sin(math.radians(el))


# --- load OSM -------------------------------------------------------------
def load_osm():
    data = json.load(open(OSM_FILE))
    buildings, trees, tree_rows, roads = [], [], [], []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if "building" in tags:
            if el["type"] == "way" and "geometry" in el:
                ring = [(g["lat"], g["lon"]) for g in el["geometry"]]
                buildings.append({"rings": [("outer", ring)], "tags": tags})
            elif el["type"] == "relation":
                rings = [(m.get("role", "outer"),
                          [(g["lat"], g["lon"]) for g in m["geometry"]])
                         for m in el.get("members", []) if "geometry" in m]
                if rings:
                    buildings.append({"rings": rings, "tags": tags})
        elif tags.get("natural") == "tree":
            trees.append({"lat": el["lat"], "lon": el["lon"], "tags": tags})
        elif tags.get("natural") == "tree_row" and "geometry" in el:
            tree_rows.append([(g["lat"], g["lon"]) for g in el["geometry"]])
        elif "highway" in tags and "geometry" in el:
            roads.append({"line": [(g["lat"], g["lon"]) for g in el["geometry"]],
                          "highway": tags["highway"]})
    return buildings, trees, tree_rows, roads


def fallback_height(tags):
    h = parse_num(tags.get("height"))
    if h and h > 2:
        return h
    lv = parse_num(tags.get("building:levels"))
    if lv and lv > 0:
        per = LEVEL_H_INSTITUTIONAL if tags.get("building") in INSTITUTIONAL else LEVEL_H
        return lv * per + 1.0
    return BUILDING_DEFAULT_H.get(tags.get("building"), 10.0)


# --- DBT integration --------------------------------------------------------
def dbt_heights(buildings, utm2xy):
    """Assign DBT-derived heights to OSM buildings; return extra DBT-only prisms."""
    from matplotlib.path import Path
    dbt = json.load(open(DBT_FILE))

    gx, gy = utm2xy([p[0] for p in dbt["ground_pts"]],
                    [p[1] for p in dbt["ground_pts"]])
    gz = np.array([p[2] for p in dbt["ground_pts"]])
    ground_med = float(np.median(gz))

    rx, ry = utm2xy([p[0] for p in dbt["roof_pts"]],
                    [p[1] for p in dbt["roof_pts"]])
    rz = np.array([p[2] for p in dbt["roof_pts"]])
    roof_xy = np.column_stack([rx, ry])

    rings = []
    for r in dbt["rings"]:
        x, y = utm2xy([p[0] for p in r["pts"]], [p[1] for p in r["pts"]])
        z = float(np.median([p[2] for p in r["pts"]]))
        rings.append({"xy": np.column_stack([x, y]), "z": z,
                      "cx": float(x.mean()), "cy": float(y.mean()),
                      "matched": False})
    ring_cxy = np.array([[r["cx"], r["cy"]] for r in rings])
    ring_z = np.array([r["z"] for r in rings])

    def local_ground(x, y):
        d2 = (gx - x) ** 2 + (gy - y) ** 2
        near = d2 <= 40.0 ** 2
        return float(np.median(gz[near])) if near.sum() >= 3 else ground_med

    n_dbt = 0
    for b in buildings:
        zs = []
        for role, ring in b["rings"]:
            if role != "outer":
                continue
            poly = np.array([to_xy(la, lo) for la, lo in ring])
            path = Path(poly)
            bb_min, bb_max = poly.min(0) - 1, poly.max(0) + 1
            pre = np.all((ring_cxy >= bb_min) & (ring_cxy <= bb_max), axis=1)
            if pre.any():
                inside = path.contains_points(ring_cxy[pre])
                idx = np.where(pre)[0][inside]
                zs += list(ring_z[idx])
                for i in idx:
                    rings[i]["matched"] = True
            pre = np.all((roof_xy >= bb_min) & (roof_xy <= bb_max), axis=1)
            if pre.any():
                inside = path.contains_points(roof_xy[pre])
                zs += list(rz[np.where(pre)[0][inside]])
        if zs:
            x, y = np.mean([to_xy(la, lo) for la, lo in b["rings"][0][1]], axis=0)
            h = float(np.percentile(zs, 75)) - local_ground(x, y)
            b["height"] = float(np.clip(h, 2.5, 80.0))
            b["height_src"] = "dbt"
            n_dbt += 1
        else:
            b["height"] = fallback_height(b["tags"])
            b["height_src"] = "fallback"

    extras = []
    for r in rings:
        if not r["matched"]:
            h = r["z"] - local_ground(r["cx"], r["cy"])
            if h >= 2.5:
                extras.append({"xy": r["xy"], "height": float(min(h, 80.0))})
    print(f"DBT heights: {n_dbt}/{len(buildings)} OSM buildings matched, "
          f"{len(extras)} DBT-only volumes added, "
          f"ground median {ground_med:.1f} m a.s.l.")

    tx, ty = utm2xy([p[0] for p in dbt["trees"]], [p[1] for p in dbt["trees"]])
    return extras, np.column_stack([tx, ty])


# --- trees: OSM + municipality + DBT, deduplicated --------------------------
class SpatialIndex:
    def __init__(self, cell=8.0):
        self.cell = cell
        self.d = {}

    def _key(self, x, y):
        return (int(x // self.cell), int(y // self.cell))

    def has_within(self, x, y, r):
        kx, ky = self._key(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in self.d.get((kx + dx, ky + dy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 <= r * r:
                        return True
        return False

    def add(self, x, y):
        self.d.setdefault(self._key(x, y), []).append((x, y))


def build_tree_list(osm_trees, tree_rows, dbt_tree_xy=None):
    muni = json.load(open(MUNI_FILE))["features"]
    mx, my, mh, mc = [], [], [], []
    for ft in muni:
        lon, lat = ft["geometry"]["coordinates"][:2]
        x, y = to_xy(lat, lon)
        h = parse_num(ft["properties"].get("h_m"))
        c = parse_num(ft["properties"].get("diam_chiom"))
        mx.append(x); my.append(y)
        mh.append(h if h and h >= 2 else np.nan)
        mc.append(c if c and c >= 1 else np.nan)
    mx, my = np.array(mx), np.array(my)
    mh, mc = np.array(mh), np.array(mc)
    med_h = float(np.nanmedian(mh))
    med_c = float(np.nanmedian(mc))

    def muni_attrs(x, y):
        d2 = (mx - x) ** 2 + (my - y) ** 2
        j = int(np.argmin(d2)) if len(mx) else -1
        if j >= 0 and d2[j] <= 8.0 ** 2:
            return j, (None if np.isnan(mh[j]) else float(mh[j])), \
                   (None if np.isnan(mc[j]) else float(mc[j]))
        return -1, None, None

    trees, idx = [], SpatialIndex()
    matched = np.zeros(len(mx), bool)

    def add(x, y, h, c, evergreen, min_dist):
        if idx.has_within(x, y, min_dist):
            return False
        trees.append((x, y, h or med_h, c or med_c, evergreen))
        idx.add(x, y)
        return True

    for t in osm_trees:
        x, y = to_xy(t["lat"], t["lon"])
        h = parse_num(t["tags"].get("height"))
        c = parse_num(t["tags"].get("diameter_crown") or t["tags"].get("crown_diameter"))
        j, mh_, mc_ = muni_attrs(x, y)
        if j >= 0:
            matched[j] = True
            h, c = h or mh_, c or mc_
        add(x, y, h, c, t["tags"].get("leaf_cycle") == "evergreen", 0.0)
    n_osm = len(trees)

    n_muni = 0
    for j in range(len(mx)):
        if not matched[j]:
            h = None if np.isnan(mh[j]) else float(mh[j])
            c = None if np.isnan(mc[j]) else float(mc[j])
            n_muni += add(float(mx[j]), float(my[j]), h, c, False, 4.0)

    n_row = 0
    for line in tree_rows:
        pts = [to_xy(la, lo) for la, lo in line]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            seg = math.hypot(x2 - x1, y2 - y1)
            for s in np.arange(0, seg, 4.0):
                f = s / seg
                n_row += add(x1 + f * (x2 - x1), y1 + f * (y2 - y1),
                             None, None, False, 3.5)

    n_dbt = 0
    if dbt_tree_xy is not None:
        for x, y in dbt_tree_xy:
            j, h, c = muni_attrs(x, y)
            n_dbt += add(float(x), float(y), h, c, False, 8.0)

    print(f"trees: {n_osm} OSM + {n_muni} municipality + {n_row} row stems "
          f"+ {n_dbt} DBT = {len(trees)} (medians h={med_h:.0f} m c={med_c:.0f} m)")
    return trees


# --- rasterisation ----------------------------------------------------------
class Grid:
    def __init__(self, xmin, xmax, ymin, ymax, res):
        self.res = res
        self.xmin, self.ymax = xmin, ymax
        self.nx = int(math.ceil((xmax - xmin) / res))
        self.ny = int(math.ceil((ymax - ymin) / res))

    def cell(self, x, y):
        return int((self.ymax - y) / self.res), int((x - self.xmin) / self.res)

    def centers(self):
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.res
        ys = self.ymax - (np.arange(self.ny) + 0.5) * self.res
        return xs, ys


def polygon_cells(grid, ring_xy):
    from matplotlib.path import Path
    ring_xy = np.asarray(ring_xy)
    i0, j0 = grid.cell(ring_xy[:, 0].min(), ring_xy[:, 1].max())
    i1, j1 = grid.cell(ring_xy[:, 0].max(), ring_xy[:, 1].min())
    i0, i1 = max(i0, 0), min(i1 + 1, grid.ny)
    j0, j1 = max(j0, 0), min(j1 + 1, grid.nx)
    if i0 >= i1 or j0 >= j1:
        return None
    xs, ys = grid.centers()
    xx, yy = np.meshgrid(xs[j0:j1], ys[i0:i1])
    inside = Path(ring_xy).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return (slice(i0, i1), slice(j0, j1)), inside


def stamp_crown(grid, top, bot, x, y, r, h, base):
    """Ellipsoidal crown: half-thickness shrinks toward the crown edge."""
    i0, j0 = grid.cell(x - r, y + r)
    i1, j1 = grid.cell(x + r, y - r)
    i0, i1 = max(i0, 0), min(i1 + 1, grid.ny)
    j0, j1 = max(j0, 0), min(j1 + 1, grid.nx)
    if i0 >= i1 or j0 >= j1:
        return
    xs, ys = grid.centers()
    xx, yy = np.meshgrid(xs[j0:j1], ys[i0:i1])
    rr2 = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r)
    inside = rr2 <= 1.0
    if not inside.any():
        return
    zc, b2 = (h + base) / 2.0, (h - base) / 2.0
    t = b2 * np.sqrt(np.clip(1.0 - rr2[inside], 0, None))
    sub_t = top[i0:i1, j0:j1]; sub_b = bot[i0:i1, j0:j1]
    sub_t[inside] = np.maximum(sub_t[inside], zc + t)
    sub_b[inside] = np.minimum(sub_b[inside], zc - t)


def build_rasters(buildings, extras, trees):
    from campus_config import FETCH_BBOX
    x0, y0 = to_xy(FETCH_BBOX[0], FETCH_BBOX[1])
    x1, y1 = to_xy(FETCH_BBOX[2], FETCH_BBOX[3])
    grid = Grid(x0, x1, y0, y1, RES)
    print(f"grid: {grid.ny} x {grid.nx} cells @ {RES} m")

    shape = (grid.ny, grid.nx)
    bld = np.zeros(shape, np.float32)
    etop = np.zeros(shape, np.float32); ebot = np.full(shape, np.inf, np.float32)
    dtop = np.zeros(shape, np.float32); dbot = np.full(shape, np.inf, np.float32)

    for x, y, h, crown, evergreen in trees:
        h = min(max(h, 3.0), 30.0)
        r = min(max(crown / 2.0, 1.0), 8.0)
        base = max(2.0, 0.35 * h)
        if evergreen:
            stamp_crown(grid, etop, ebot, x, y, r, h, base)
        else:
            stamp_crown(grid, dtop, dbot, x, y, r, h, base)

    relations = [b for b in buildings if len(b["rings"]) > 1
                 or b["rings"][0][0] == "inner"]
    simple = [b for b in buildings if b not in relations]
    for b in relations:
        for role, ring in b["rings"]:
            cells = polygon_cells(grid, [to_xy(la, lo) for la, lo in ring])
            if cells is None:
                continue
            sl, inside = cells
            if role == "outer":
                bld[sl][inside] = np.maximum(bld[sl][inside], b["height"])
            else:
                bld[sl][inside] = 0.0
    for b in simple:
        cells = polygon_cells(grid, [to_xy(la, lo) for la, lo in b["rings"][0][1]])
        if cells is not None:
            sl, inside = cells
            bld[sl][inside] = np.maximum(bld[sl][inside], b["height"])
    for e in extras:
        cells = polygon_cells(grid, e["xy"])
        if cells is not None:
            sl, inside = cells
            bld[sl][inside] = np.maximum(bld[sl][inside], e["height"])

    ebot[np.isinf(ebot)] = 0.0
    dbot[np.isinf(dbot)] = 0.0
    return grid, bld, (etop, ebot), (dtop, dbot)


def build_cover(grid, roads, bmask_full):
    """uint8 ground-cover class raster over the full grid."""
    cover = np.full((grid.ny, grid.nx), COVER["paved"], np.uint8)
    data = json.load(open(COVER_FILE))

    def cls_for(tags):
        lu = tags.get("landuse"); le = tags.get("leisure")
        na = tags.get("natural"); am = tags.get("amenity")
        if na == "water" or le == "swimming_pool" or lu == "basin":
            return "water"
        if le == "pitch" or le == "track":
            return "sport"
        if lu in {"grass", "meadow", "village_green", "flowerbed", "forest",
                  "recreation_ground", "cemetery", "greenfield", "vineyard",
                  "orchard", "allotments"} \
           or le in {"park", "garden", "dog_park", "playground", "golf_course"} \
           or na in {"wood", "scrub", "grassland"}:
            return "grass"
        if am == "parking":
            return "asphalt"
        return None

    layers = {"grass": [], "sport": [], "asphalt": [], "water": []}
    for el in data["elements"]:
        if "geometry" not in el:
            continue
        c = cls_for(el.get("tags", {}))
        if c:
            layers[c].append([to_xy(g["lat"], g["lon"]) for g in el["geometry"]])
    for cname in ("grass", "sport", "asphalt", "water"):   # later wins
        for ring in layers[cname]:
            cells = polygon_cells(grid, ring)
            if cells is not None:
                sl, inside = cells
                cover[sl][inside] = COVER[cname]

    # roads stamped over areas (bridge decks, lanes through parks...)
    xs, ys = grid.centers()
    for rd in roads:
        w = ROAD_WIDTH.get(rd["highway"])
        if w is None:
            continue
        cls = COVER["footpath"] if rd["highway"] in PEDESTRIAN else COVER["asphalt"]
        pts = [to_xy(la, lo) for la, lo in rd["line"]]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            seg = math.hypot(xb - xa, yb - ya)
            for s in np.arange(0, seg + RES / 2, RES):
                f = min(s / seg, 1.0) if seg else 0
                x, y = xa + f * (xb - xa), ya + f * (yb - ya)
                r = w / 2.0
                i0, j0 = grid.cell(x - r, y + r)
                i1, j1 = grid.cell(x + r, y - r)
                i0, i1 = max(i0, 0), min(i1 + 1, grid.ny)
                j0, j1 = max(j0, 0), min(j1 + 1, grid.nx)
                if i0 >= i1 or j0 >= j1:
                    continue
                xx, yy = np.meshgrid(xs[j0:j1], ys[i0:i1])
                disc = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
                cover[i0:i1, j0:j1][disc] = cls

    cover[bmask_full] = COVER["building"]
    return cover


# --- shadow / sun fraction ---------------------------------------------------
def sun_fraction(BLD, ET, EB, DT, DB, rec, elev, az, hmax, tau_decid):
    """Fraction of direct beam reaching the ground, per receptor cell."""
    r0, r1, c0, c1 = rec
    tanel = math.tan(math.radians(elev))
    dx, dy = math.sin(math.radians(az)), math.cos(math.radians(az))
    shape = (r1 - r0, c1 - c0)
    bhit = np.zeros(shape, bool)
    ehit = np.zeros(shape, bool)
    dhit = np.zeros(shape, bool)
    seen = set()
    for k in range(1, KCAP + 1):
        if k * RES * tanel > hmax:
            break
        ox, oy = round(k * dx), round(-k * dy)
        if (ox, oy) in seen:
            continue
        seen.add((ox, oy))
        rayh = math.hypot(ox, oy) * RES * tanel
        sl = (slice(r0 + oy, r1 + oy), slice(c0 + ox, c1 + ox))
        bhit |= BLD[sl] > rayh + 0.01
        ehit |= (ET[sl] > rayh + 0.01) & (EB[sl] < rayh)
        dhit |= (DT[sl] > rayh + 0.01) & (DB[sl] < rayh)
    f = (~bhit).astype(np.float32)
    f *= np.where(ehit, TAU_LEAF, 1.0)
    f *= np.where(dhit, tau_decid, 1.0)
    return f


def main():
    buildings, osm_trees, tree_rows, roads = load_osm()
    print(f"OSM: {len(buildings)} buildings, {len(osm_trees)} trees, "
          f"{len(tree_rows)} tree rows, {len(roads)} road segments")

    utm2xy = make_utm_to_xy()
    extras, dbt_tree_xy = dbt_heights(buildings, utm2xy)
    trees = build_tree_list(osm_trees, tree_rows, dbt_tree_xy)

    # optional intervention scenario: extra trees / shade sails as obstacles
    import os
    out_file = OUT_FILE
    scen_path = os.environ.get("SCENARIO")
    if scen_path:
        sc = json.load(open(scen_path))
        for x, y, h, crown in sc.get("trees", []):
            trees.append((x, y, h, crown, False))
        for x, y in sc.get("sails", []):
            # 6x6 m tensile sail at ~4 m: modeled as a flat "evergreen crown"
            # of equal area (r 3.4 m, tau 0.15 vs the real ~0.10)
            trees.append((x, y, 4.1, 6.8, True))
        out_file = OUT_FILE.replace(".npz", f"_{sc['name']}.npz")
        print(f"scenario '{sc['name']}': +{len(sc.get('trees', []))} trees, "
              f"+{len(sc.get('sails', []))} sails -> {out_file}")

    grid, bld, (etop, ebot), (dtop, dbot) = build_rasters(buildings, extras, trees)
    bmask_full = bld > 0
    cover = build_cover(grid, roads, bmask_full)

    pad = KCAP + 2
    P = lambda a: np.pad(a, pad)
    BLD, ET, EB, DT, DB = P(bld), P(etop), P(ebot), P(dtop), P(dbot)

    xw, yn = to_xy(REC_N, REC_W)
    xe, ys = to_xy(REC_S, REC_E)
    i0, j0 = grid.cell(xw, yn)
    i1, j1 = grid.cell(xe, ys)
    rec = (i0 + pad, i1 + pad, j0 + pad, j1 + pad)
    rec_shape = (i1 - i0, j1 - j0)
    bmask_rec = bmask_full[i0:i1, j0:j1]
    cover_rec = cover[i0:i1, j0:j1]
    hmax = float(max(BLD.max(), ET.max(), DT.max()))
    print(f"receptor grid: {rec_shape[0]} x {rec_shape[1]}, hmax {hmax:.1f} m")

    minutes = np.arange(3 * 60, 20 * 60, STEP_MIN)  # UTC
    hours = np.zeros((12, *rec_shape), np.float32)      # direct-sun h/day
    insol = np.zeros((12, *rec_shape), np.float32)      # kWh/m2/day
    midday = np.zeros((12, *rec_shape), np.float32)     # sunny minutes 12-14 local
    daylight = np.zeros(12, np.float32)

    import time
    for m in range(1, 13):
        t0 = time.time()
        tau_d = TAU_BARE if m in LEAF_OFF_MONTHS else TAU_LEAF
        utc_off = 2 if 4 <= m <= 10 else 1              # CEST / CET
        win = ((12 - utc_off) * 60, (14 - utc_off) * 60)
        jd = julian_date(2025, m, SAMPLE_DAY, minutes.astype(float))
        elev, az = solar_position(jd, LAT0, LON0)
        keep = elev > MIN_ELEV
        for mins, e, a in zip(minutes[keep], elev[keep], az[keep]):
            f = sun_fraction(BLD, ET, EB, DT, DB, rec, e, a, hmax, tau_d)
            fbin = f >= 0.5
            hours[m - 1] += fbin * (STEP_MIN / 60.0)
            insol[m - 1] += f * clear_sky_direct_horizontal(e) \
                * (STEP_MIN / 60.0) / 1000.0
            if win[0] <= mins < win[1]:
                midday[m - 1] += fbin * STEP_MIN
        daylight[m - 1] = keep.sum() * STEP_MIN / 60.0
        print(f"month {m:2d}: daylight {daylight[m-1]:4.1f} h, "
              f"tau_decid {tau_d:.2f}, {time.time()-t0:.1f}s")

    annual_h = hours.mean(axis=0)
    annual_i = insol.mean(axis=0)
    annual_m = midday.mean(axis=0)

    xs, ys_c = grid.centers()
    xs = xs[j0:j1]; ys_c = ys_c[i0:i1]
    xx, yy = np.meshgrid(xs, ys_c)
    season_idx = {"DJF": [11, 0, 1], "MAM": [2, 3, 4],
                  "JJA": [5, 6, 7], "SON": [8, 9, 10]}
    print("\n--- POI results (mean over 15 m disc, ground cells only) ---")
    poi_out = {}
    for name, (la, lo) in POIS.items():
        x, y = to_xy(la, lo)
        sel = ((xx - x) ** 2 + (yy - y) ** 2 <= POI_RADIUS ** 2) & ~bmask_rec
        entry = {
            "annual_hours": float(annual_h[sel].mean()),
            "annual_insol": float(annual_i[sel].mean()),
            "annual_midday_min": float(annual_m[sel].mean()),
            "monthly_hours": [float(hours[m][sel].mean()) for m in range(12)],
            "monthly_insol": [float(insol[m][sel].mean()) for m in range(12)],
            "monthly_midday": [float(midday[m][sel].mean()) for m in range(12)],
        }
        for s, ms in season_idx.items():
            entry[f"hours_{s}"] = float(np.mean([hours[m][sel].mean() for m in ms]))
        poi_out[name] = entry
        print(f"{name}:")
        print(f"   sun {entry['annual_hours']:.2f} h/day | direct insolation "
              f"{entry['annual_insol']:.2f} kWh/m2/day | lunch window "
              f"{entry['annual_midday_min']:.0f}/120 min in sun")
        print("   seasons h/day: " + "  ".join(
            f"{s} {entry['hours_' + s]:.1f}" for s in season_idx))

    np.savez_compressed(
        out_file, hours=hours, insol=insol, midday=midday,
        annual_hours=annual_h, annual_insol=annual_i, annual_midday=annual_m,
        bmask=bmask_rec, cover=cover_rec, daylight=daylight,
        trees=np.array([(x, y, min(max(c / 2, 1), 8)) for x, y, h, c, _ in trees],
                       np.float32),
        extent=np.array([xs[0] - RES / 2, xs[-1] + RES / 2,
                         ys_c[-1] - RES / 2, ys_c[0] + RES / 2]),
        origin=np.array([LAT0, LON0]), res=np.array([RES]),
        poi=json.dumps(poi_out))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
