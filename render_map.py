"""Render sun-exposure and heat maps: annual, 4 seasons, lunch window, heat proxy."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from campus_config import CFG, SUN_FILE as NPZ, OUT_DIR, is_polimi
from compute_sun_hours import load_osm, to_xy, POIS, COVER

LABEL = CFG["label"]

# single-hue ramps (validated: monotone lightness, one hue)
SUN_RAMP = ["#fdf6e3", "#f6d98f", "#eeb63e", "#dd9500", "#b17300", "#865300", "#5e3800"]
HEAT_RAMP = ["#fdf0ec", "#f6c9b8", "#ee9f83", "#e37450", "#c94e2c", "#a03418", "#711f0c"]
INK = "#3a3a37"
MUTED = "#6f6e6a"
BUILDING_FILL = "#d7d5d0"
BUILDING_EDGE = "#b9b7b1"
POLIMI_FILL = "#c3dcef"   # campus buildings pop out light-blue
POLIMI_EDGE = "#7fa8c9"
TREE_EDGE = "#4c7a4c"

# ground-cover -> solar absorption/heat factor (proxy: 1-albedo, discounted
# for evapotranspiration on vegetation and water)
HEAT_K = {COVER["paved"]: 0.75, COVER["asphalt"]: 0.92, COVER["footpath"]: 0.70,
          COVER["grass"]: 0.35, COVER["water"]: 0.20, COVER["sport"]: 0.80}

SEASONS = [("Winter (DJF)", [11, 0, 1]), ("Spring (MAM)", [2, 3, 4]),
           ("Summer (JJA)", [5, 6, 7]), ("Autumn (SON)", [8, 9, 10])]


def halo(lw=3):
    return [pe.withStroke(linewidth=lw, foreground="white")]


EXCL_XY = [(to_xy(la, lo), r) for la, lo, r in CFG.get("exclude_dest", [])]


def is_campus(b):
    """PoliMi building, excluding the office-only sites in exclude_dest."""
    if not is_polimi(b["tags"]):
        return False
    ring = b["rings"][0][1]
    la = sum(p[0] for p in ring) / len(ring)
    lo = sum(p[1] for p in ring) / len(ring)
    x, y = to_xy(la, lo)
    return not any((x - ex) ** 2 + (y - ey) ** 2 < r * r
                   for (ex, ey), r in EXCL_XY)


def polimi_mask(d, buildings, extent):
    """Grid mask of campus-building cells (same courtyard topology as bmask)."""
    from matplotlib.path import Path
    bmask = d["bmask"]
    ny, nx = bmask.shape
    x0, x1, y0, y1 = extent
    res = float(d["res"][0])
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y1 - (np.arange(ny) + 0.5) * res
    mask = np.zeros_like(bmask)
    for b in buildings:
        if not is_campus(b):
            continue
        for role, ring in sorted(b["rings"], key=lambda r: r[0] == "inner"):
            xy = np.array([to_xy(la, lo) for la, lo in ring])
            j0 = max(0, int((xy[:, 0].min() - x0) / res) - 1)
            j1 = min(nx, int((xy[:, 0].max() - x0) / res) + 2)
            i0 = max(0, int((y1 - xy[:, 1].max()) / res) - 1)
            i1 = min(ny, int((y1 - xy[:, 1].min()) / res) + 2)
            if j0 >= j1 or i0 >= i1:
                continue
            xx, yy = np.meshgrid(xs[j0:j1], ys[i0:i1])
            inside = Path(xy).contains_points(
                np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            if role == "inner":
                mask[i0:i1, j0:j1] &= ~inside
            else:
                mask[i0:i1, j0:j1] |= inside
    return mask & bmask


def draw_context(ax, d, buildings, roads, extent, show_trees=True):
    x0, x1, y0, y1 = extent
    bmask = d["bmask"]
    pmask = getattr(draw_context, "_pmask", None)
    if pmask is None:
        pmask = draw_context._pmask = polimi_mask(d, buildings, extent)
    bld_rgba = np.zeros((*bmask.shape, 4))
    bld_rgba[bmask] = to_rgba(BUILDING_FILL)
    bld_rgba[pmask] = to_rgba(POLIMI_FILL)
    ax.imshow(bld_rgba, extent=extent, interpolation="nearest", zorder=3)
    for rd in roads:
        xy = np.array([to_xy(la, lo) for la, lo in rd["line"]])
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=0.9, alpha=0.3,
                solid_capstyle="round", zorder=2)
    for b in buildings:
        edge = POLIMI_EDGE if is_campus(b) else BUILDING_EDGE
        for role, ring in b["rings"]:
            xy = np.array([to_xy(la, lo) for la, lo in ring])
            ax.plot(xy[:, 0], xy[:, 1], color=edge,
                    lw=0.7 if edge is POLIMI_EDGE else 0.4, zorder=4)
    if show_trees:
        t = d["trees"]
        sel = (t[:, 0] > x0 - 10) & (t[:, 0] < x1 + 10) & \
              (t[:, 1] > y0 - 10) & (t[:, 1] < y1 + 10)
        t = t[sel]
        ec = EllipseCollection(t[:, 2] * 2, t[:, 2] * 2, 0, units="xy",
                               offsets=t[:, :2], transOffset=ax.transData,
                               facecolors="none", edgecolors=TREE_EDGE,
                               linewidths=0.3, alpha=0.35, zorder=5)
        ax.add_collection(ec)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_pois(ax, values=None, fontsize=10.5):
    for name, (la, lo) in POIS.items():
        x, y = to_xy(la, lo)
        ax.plot(x, y, "o", ms=8, mfc=INK, mec="white", mew=1.6, zorder=7)
        tag = name.split()[0]
        label = tag if values is None else f"{tag} · {values[name]}"
        ax.annotate(label, (x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=fontsize, color=INK, fontweight="bold", zorder=7,
                    path_effects=halo())


def draw_scalebar(ax, extent):
    x0, x1, y0, y1 = extent
    bx, by = x0 + 25, y0 + 22
    ax.plot([bx, bx + 100], [by, by], color=INK, lw=2.5,
            solid_capstyle="butt", zorder=7)
    ax.annotate("100 m", (bx + 50, by), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, color=INK, zorder=7, path_effects=halo())


def _conv_same(a, k):
    H, W = a.shape
    kh, kw = k.shape
    s = (H + kh - 1, W + kw - 1)
    out = np.fft.irfft2(np.fft.rfft2(a, s) * np.fft.rfft2(k, s), s)
    return out[kh // 2:kh // 2 + H, kw // 2:kw // 2 + W]


def microclimate(absorbed, ground, res, sigma_m=10.0):
    """What a spot feels depends on its surroundings (a stone path between
    lawns is cooler than the same stone in an asphalt expanse): normalized
    Gaussian mixing of absorbed solar over ground cells, ~sigma_m reach."""
    r = max(1, int(3 * sigma_m / res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx * xx + yy * yy) * (res * res) / (2 * sigma_m ** 2))
    m = ground.astype(np.float32)
    num = _conv_same(absorbed * m, g)
    den = _conv_same(m, g)
    return np.where(ground, num / np.maximum(den, 1e-9), 0.0)


def heat_field(d):
    insol_jja = d["insol"][[5, 6, 7]].mean(axis=0)
    k = np.zeros_like(insol_jja)
    for cls, kk in HEAT_K.items():
        k[d["cover"] == cls] = kk
    return microclimate(insol_jja * k, ~d["bmask"], float(d["res"][0]))


def main():
    d = np.load(NPZ, allow_pickle=False)
    extent = tuple(d["extent"])
    poi = json.loads(str(d["poi"]))
    buildings, _, _, roads = load_osm()
    sun_cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)
    heat_cmap = LinearSegmentedColormap.from_list("heat", HEAT_RAMP)

    def field(a):
        return np.ma.masked_where(d["bmask"], a)

    # --- 1. annual daily sun hours ---------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 9), dpi=200)
    vmax = math.ceil(float(d["annual_hours"].max()))
    im = ax.imshow(field(d["annual_hours"]), extent=extent, cmap=sun_cmap,
                   vmin=0, vmax=vmax, interpolation="nearest", zorder=1)
    draw_context(ax, d, buildings, roads, extent)
    draw_scalebar(ax, extent)
    cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    cb.set_label("direct sun · hours per day (annual mean)", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)
    ax.set_title("Average daily sun exposure — " + LABEL,
                 fontsize=15, color=INK, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.012, "OSM footprints + Milano DBT surveyed heights · semi-transparent "
            "ellipsoid crowns (τ 0.15 leafed / 0.55 bare) · light-blue = campus buildings · gray = other buildings",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(f"{OUT_DIR}/sun_map_annual.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/sun_map_annual.png")

    # --- 2. four seasons ---------------------------------------------------
    sfields = {name: d["hours"][ms].mean(axis=0) for name, ms in SEASONS}
    vmax_s = math.ceil(max(float(f.max()) for f in sfields.values()))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11), dpi=170)
    for ax, (name, ms) in zip(axes.ravel(), SEASONS):
        im = ax.imshow(field(sfields[name]), extent=extent, cmap=sun_cmap,
                       vmin=0, vmax=vmax_s, interpolation="nearest", zorder=1)
        draw_context(ax, d, buildings, roads, extent, show_trees=False)
        ax.set_title(name, fontsize=12, color=INK, loc="left", fontweight="bold")
    draw_scalebar(axes[1, 0], extent)
    fig.subplots_adjust(hspace=0.08, wspace=0.04, right=0.88)
    cax = fig.add_axes([0.90, 0.25, 0.018, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("direct sun · hours per day", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)
    fig.suptitle("Seasonal sun exposure — " + LABEL,
                 fontsize=15, color=INK, x=0.095, ha="left", fontweight="bold")
    fig.savefig(f"{OUT_DIR}/sun_map_seasons.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/sun_map_seasons.png")

    # --- 3. lunch window (12-14 local), summer ----------------------------
    mid_jja = d["midday"][[5, 6, 7]].mean(axis=0)
    fig, ax = plt.subplots(figsize=(11.5, 9), dpi=200)
    im = ax.imshow(field(mid_jja), extent=extent, cmap=sun_cmap,
                   vmin=0, vmax=120, interpolation="nearest", zorder=1)
    draw_context(ax, d, buildings, roads, extent)
    draw_scalebar(ax, extent)
    cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    cb.set_label("minutes in direct sun during 12–14 (of 120)", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)
    ax.set_title("Lunch-walk exposure, summer — " + LABEL,
                 fontsize=15, color=INK, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.012, "June–August mean · 12:00–14:00 local time, when people cross "
            "campus to eat · light-blue = campus buildings · gray = other buildings",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(f"{OUT_DIR}/sun_map_midday.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/sun_map_midday.png")

    # --- 4. summer surface heat proxy --------------------------------------
    heat = heat_field(d)
    fig, ax = plt.subplots(figsize=(11.5, 9), dpi=200)
    vmax_h = math.ceil(float(heat.max()) * 2) / 2
    im = ax.imshow(field(heat), extent=extent, cmap=heat_cmap,
                   vmin=0, vmax=vmax_h, interpolation="nearest", zorder=1)
    draw_context(ax, d, buildings, roads, extent)
    draw_scalebar(ax, extent)
    cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    cb.set_label("absorbed direct solar energy · kWh/m²/day (summer)",
                 fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)
    ax.set_title("Summer surface heat load (proxy) — " + LABEL,
                 fontsize=15, color=INK, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.012, "June–August direct insolation × surface absorption "
            "(asphalt 0.92 · paved 0.75 · sport 0.80 · grass 0.35 · water 0.20) · "
            "10 m neighbourhood mixing · DBT surveyed surfaces",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(f"{OUT_DIR}/heat_map_summer.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/heat_map_summer.png")


if __name__ == "__main__":
    main()
