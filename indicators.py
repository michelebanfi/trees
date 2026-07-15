"""Quantitative indicators for the shading proposal: baseline vs the two
equal-budget solutions (trees / shade sails).

Computed (qualitative indicators are out of scope):
  - % of inter-building pedestrian routes in shade (JJA)
  - % of campus area under shade (JJA)
  - average surface temperature (deg C, JJA afternoon proxy)
  - number of trees / green area (m2) / new shade canopy (m2)
  - implementation cost, annual maintenance, implementation time (constants
    in scenarios.py — rough guesses to be refined)

Also evaluates the proposal's example points (campus_config proposal_points)
and renders before/after zoom maps for each.

Run AFTER scenarios.py and the two SCENARIO= re-simulations.
Outputs: output/<campus>/indicators.md, proposal_points.png
"""
import json
import math

import numpy as np

from campus_config import CFG, SUN_FILE, FLOWS_FILE, DATA_DIR, OUT_DIR
from compute_sun_hours import to_xy, load_osm, COVER
from render_map import polimi_mask, HEAT_K
from route_flows import (load_transit, classify_stops, build_destinations,
                         build_graph, dijkstra, nearest_node)
from scenarios import (dilate, Field, CAMPUS_ZONE_M, BUDGET,
                       TREE_COST, TREE_MAINT, TREE_TIME_Y, TREE_CROWN,
                       SAIL_COST, SAIL_MAINT, SAIL_TIME_Y, SAIL_SIDE)

SHADE_H = 4.0     # JJA direct sun below this (h/day) counts as "in shade"
T_AIR = 26.0      # typical Milan JJA afternoon air temperature (deg C)
T_COEF = 4.3      # deg C per kWh/m2/day of absorbed direct solar (proxy fit)
POINT_R = 20.0    # evaluation disc around each proposal point (m)

SCENARIOS = ["baseline", "trees", "sails"]


def npz_path(scen):
    return SUN_FILE if scen == "baseline" else \
        SUN_FILE.replace(".npz", f"_{scen}.npz")


class Raster:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.jja_h = d["hours"][5:8].mean(axis=0)
        self.jja_i = d["insol"][5:8].mean(axis=0)
        self.jja_mid = d["midday"][5:8].mean(axis=0)
        self.bmask = d["bmask"]
        self.cover = d["cover"]
        self.extent = d["extent"]
        self.res = float(d["res"][0])
        self.fld = Field(d)

    def sample_h(self, xy):
        i, j = self.fld.cell(*xy)
        if not self.fld.inside(i, j) or self.bmask[i, j]:
            return None
        return float(self.jja_h[i, j])


def interbuilding_usage():
    """Edge usage of building-to-building shortest walks (weight wi*wj)."""
    nodes, ways, _ = load_transit()
    _, _, _, _, _, bldg_entr = classify_stops(nodes)
    base = Raster(npz_path("baseline"))
    dests = build_destinations(bldg_entr, tuple(base.extent))
    adj, coord, edges = build_graph(ways)
    node_ids = list(adj.keys())
    node_arr = np.array([coord[n] for n in node_ids])
    for d in dests:
        d["node"] = nearest_node(d["doors"][0], node_arr, node_ids)

    usage = {}
    for a in range(len(dests)):
        src = dests[a]["node"]
        distm, prev = dijkstra(adj, src)
        for b in range(a + 1, len(dests)):
            tgt = dests[b]["node"]
            if tgt not in distm or tgt == src:
                continue
            w = dests[a]["w"] * dests[b]["w"]
            n = tgt
            while n != src:
                pn, key = prev[n]
                usage[key] = usage.get(key, 0.0) + w
                n = pn
    return usage, edges, coord


def routes_in_shade(usage, edges, coord, rasters):
    """Usage-weighted % of route meters in shade + mean sun, per scenario."""
    pct, mean_h = {}, {}
    for scen, R in rasters.items():
        num = den = hsum = 0.0
        for key, u in usage.items():
            e = edges[key]
            a, b = coord[e["u"]], coord[e["v"]]
            if e["covered"]:
                h = 0.0
            else:
                n_s = max(2, int(e["len"] / 2.0) + 1)
                s = [R.sample_h((a[0] + (b[0] - a[0]) * t,
                                 a[1] + (b[1] - a[1]) * t))
                     for t in np.linspace(0, 1, n_s)]
                s = [v for v in s if v is not None]
                if not s:
                    continue
                h = float(np.mean(s))
            den += u * e["len"]
            num += u * e["len"] * (h < SHADE_H)
            hsum += u * e["len"] * h
        pct[scen] = 100.0 * num / den
        mean_h[scen] = hsum / den
    return pct, mean_h


def arrival_exposure(rasters):
    """Flow-weighted JJA sun (h/day) on the meters PT arrivals walk."""
    fl = json.load(open(FLOWS_FILE))
    out = {}
    for scen, R in rasters.items():
        num = den = 0.0
        for e in fl["edges"]:
            (ax, ay), (bx, by) = e["a"], e["b"]
            n_s = max(2, int(e["len"] / 2.0) + 1)
            s = [R.sample_h((ax + (bx - ax) * t, ay + (by - ay) * t))
                 for t in np.linspace(0, 1, n_s)]
            s = [v for v in s if v is not None]
            if not s:
                continue
            h = 0.0 if e.get("covered") else float(np.mean(s))
            w = e["flow"] * e["len"]
            den += w
            num += w * h
        out[scen] = num / den
    return out


def surface_temp(R, zone):
    k = np.zeros_like(R.jja_i)
    for cls, kk in HEAT_K.items():
        k[R.cover == cls] = kk
    T = T_AIR + T_COEF * k * R.jja_i
    return float(T[zone].mean())


def main():
    rasters = {s: Raster(npz_path(s)) for s in SCENARIOS}
    base = rasters["baseline"]
    d0 = np.load(npz_path("baseline"), allow_pickle=False)
    buildings, _, _, _ = load_osm()
    pmask = polimi_mask(d0, buildings, tuple(base.extent))
    zone = dilate(pmask, CAMPUS_ZONE_M) & ~base.bmask

    scen_pts = {"trees": json.load(open(f"{DATA_DIR}/scenario_trees.json")),
                "sails": json.load(open(f"{DATA_DIR}/scenario_sails.json"))}
    n_trees_new = len(scen_pts["trees"]["trees"])
    n_sails_new = len(scen_pts["sails"]["sails"])

    # ---- campus-wide indicators -----------------------------------------
    usage, edges, coord = interbuilding_usage()
    shade_routes, route_h = routes_in_shade(usage, edges, coord, rasters)
    arrivals_h = arrival_exposure(rasters)
    shade_area = {s: 100.0 * float((R.jja_h[zone] < SHADE_H).mean())
                  for s, R in rasters.items()}
    zone_h = {s: float(R.jja_h[zone].mean()) for s, R in rasters.items()}
    temp = {s: surface_temp(R, zone) for s, R in rasters.items()}

    trees_base = sum(1 for x, y, r in d0["trees"]
                     if base.fld.inside(*base.fld.cell(x, y))
                     and zone[base.fld.cell(x, y)])
    green_m2 = float((base.cover[zone] == COVER["grass"]).sum()) * base.res**2
    canopy = {"baseline": 0.0,
              "trees": n_trees_new * math.pi * (TREE_CROWN / 2) ** 2,
              "sails": n_sails_new * SAIL_SIDE ** 2}
    cost = {"baseline": 0.0, "trees": n_trees_new * TREE_COST,
            "sails": n_sails_new * SAIL_COST}
    maint = {"baseline": 0.0, "trees": n_trees_new * TREE_MAINT,
             "sails": n_sails_new * SAIL_MAINT}
    time_y = {"baseline": 0.0, "trees": TREE_TIME_Y, "sails": SAIL_TIME_Y}
    ntrees = {"baseline": trees_base, "trees": trees_base + n_trees_new,
              "sails": trees_base}

    # ---- per-point indicators -------------------------------------------
    pts = {}
    for name, (la, lo) in CFG["proposal_points"].items():
        x, y = to_xy(la, lo)
        i0, j0 = base.fld.cell(x, y)
        rr = int(POINT_R / base.res)
        ii, jj = np.mgrid[i0 - rr:i0 + rr + 1, j0 - rr:j0 + rr + 1]
        ok = ((ii - i0) ** 2 + (jj - j0) ** 2 <= rr * rr) & \
             (ii >= 0) & (ii < base.fld.ny) & (jj >= 0) & (jj < base.fld.nx)
        sel = (ii[ok], jj[ok])
        ground = ~base.bmask[sel]
        row = {}
        for s, R in rasters.items():
            k = np.zeros_like(R.jja_i)
            for cls, kk in HEAT_K.items():
                k[R.cover == cls] = kk
            row[s] = {
                "jja_h": float(R.jja_h[sel][ground].mean()),
                "lunch_min": float(R.jja_mid[sel][ground].mean()),
                "temp": float((T_AIR + T_COEF * k * R.jja_i)[sel][ground].mean()),
            }
        near = {}
        for s, sc in scen_pts.items():
            arr = sc.get("trees", sc.get("sails", []))
            near[s] = sum(1 for p in arr
                          if math.hypot(p[0] - x, p[1] - y) < 40.0)
        row["near"] = near
        pts[name] = row

    # ---- report -----------------------------------------------------------
    def fr(v, unit=""):
        return f"{v:,.1f}{unit}"

    L = []
    L.append(f"# Shading proposal indicators — {CFG['label']}\n")
    L.append(f"Equal implementation budget per solution: "
             f"**EUR {BUDGET:,.0f}** · trees scenario: {n_trees_new} "
             f"semi-mature trees · sails scenario: {n_sails_new} modules "
             f"of {SAIL_SIDE:.0f}x{SAIL_SIDE:.0f} m\n")
    L.append("'In shade' = less than "
             f"{SHADE_H:.0f} h/day of direct June-August sun. Surface "
             "temperature is a clear-sky afternoon proxy: "
             f"T = {T_AIR:.0f} degC + {T_COEF} x absorbed direct solar "
             "(kWh/m2/day). Costs are placeholder estimates.\n")
    L.append("| Indicator | Baseline | Trees | Sails |")
    L.append("|---|---|---|---|")
    L.append(f"| % inter-building routes in shade (JJA) | "
             + " | ".join(f"{shade_routes[s]:.1f}%" for s in SCENARIOS) + " |")
    L.append(f"| Mean sun on inter-building routes (h/day) | "
             + " | ".join(f"{route_h[s]:.2f}" for s in SCENARIOS) + " |")
    L.append(f"| Flow-weighted sun on arrival routes (h/day) | "
             + " | ".join(f"{arrivals_h[s]:.2f}" for s in SCENARIOS) + " |")
    L.append(f"| % campus area under shade (JJA) | "
             + " | ".join(f"{shade_area[s]:.1f}%" for s in SCENARIOS) + " |")
    L.append(f"| Mean sun over campus area (h/day) | "
             + " | ".join(f"{zone_h[s]:.2f}" for s in SCENARIOS) + " |")
    L.append(f"| Avg surface temperature (degC) | "
             + " | ".join(f"{temp[s]:.1f}" for s in SCENARIOS) + " |")
    L.append(f"| Trees on campus | "
             + " | ".join(f"{ntrees[s]:,}" for s in SCENARIOS) + " |")
    L.append(f"| Green area (m2) | " + " | ".join(
        f"{green_m2:,.0f}" for _ in SCENARIOS) + " |")
    L.append(f"| New shade canopy (m2) | " + " | ".join(
        f"{canopy[s]:,.0f}" for s in SCENARIOS) + " |")
    L.append(f"| Implementation cost (EUR) | " + " | ".join(
        f"{cost[s]:,.0f}" for s in SCENARIOS) + " |")
    L.append(f"| Annual maintenance (EUR/y) | " + " | ".join(
        f"{maint[s]:,.0f}" for s in SCENARIOS) + " |")
    L.append(f"| Implementation time (years) | " + " | ".join(
        f"{time_y[s]:.1f}" for s in SCENARIOS) + " |")
    L.append("")
    L.append("## Example points (20 m surroundings)\n")
    for name, row in pts.items():
        la, lo = CFG["proposal_points"][name]
        L.append(f"### Point {name} ({la:.6f}, {lo:.6f}) — "
                 f"{row['near']['trees']} new trees / "
                 f"{row['near']['sails']} sails within 40 m\n")
        L.append("| | Baseline | Trees | Sails |")
        L.append("|---|---|---|---|")
        L.append("| Direct sun JJA (h/day) | " + " | ".join(
            f"{row[s]['jja_h']:.1f}" for s in SCENARIOS) + " |")
        L.append("| Lunch window in sun (min/120) | " + " | ".join(
            f"{row[s]['lunch_min']:.0f}" for s in SCENARIOS) + " |")
        L.append("| Surface temperature (degC) | " + " | ".join(
            f"{row[s]['temp']:.1f}" for s in SCENARIOS) + " |")
        L.append("")
    text = "\n".join(L)
    with open(f"{OUT_DIR}/indicators.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {OUT_DIR}/indicators.md")

    render_points(rasters, scen_pts, pts)


def render_points(rasters, scen_pts, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.patches import Circle, Rectangle
    from render_map import SUN_RAMP, INK, MUTED, BUILDING_FILL

    base = rasters["baseline"]
    cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)
    Z = 55.0  # half-window (m)
    names = list(CFG["proposal_points"].keys())
    fig, axes = plt.subplots(len(names), 3, figsize=(12.5, 3.9 * len(names)),
                             dpi=170)
    fig.subplots_adjust(top=0.93, hspace=0.16, wspace=0.06)
    for r, name in enumerate(names):
        la, lo = CFG["proposal_points"][name]
        x, y = to_xy(la, lo)
        for c, scen in enumerate(SCENARIOS):
            R = rasters[scen]
            ax = axes[r, c]
            field = np.ma.masked_where(R.bmask, R.jja_h)
            ax.imshow(field, extent=R.extent, cmap=cmap, vmin=0, vmax=14,
                      interpolation="nearest", zorder=1)
            b = np.zeros((*R.bmask.shape, 4))
            b[R.bmask] = to_rgba(BUILDING_FILL)
            ax.imshow(b, extent=R.extent, interpolation="nearest", zorder=2)
            if scen != "baseline":
                arr = scen_pts[scen].get("trees",
                                         scen_pts[scen].get("sails", []))
                for p in arr:
                    if abs(p[0] - x) > Z + 8 or abs(p[1] - y) > Z + 8:
                        continue
                    if scen == "trees":
                        ax.add_patch(Circle((p[0], p[1]), TREE_CROWN / 2,
                                            fill=False, ec="#2e7d32", lw=1.6,
                                            zorder=4))
                    else:
                        ax.add_patch(Rectangle(
                            (p[0] - SAIL_SIDE / 2, p[1] - SAIL_SIDE / 2),
                            SAIL_SIDE, SAIL_SIDE, fill=False, ec="#37474f",
                            lw=1.6, zorder=4))
            ax.plot(x, y, "x", ms=10, mew=2.6, color=INK, zorder=6)
            ax.set_xlim(x - Z, x + Z); ax.set_ylim(y - Z, y + Z)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            h = pts[name][scen]["jja_h"]
            ax.set_title(f"{'Point ' + name + ' — ' if c == 0 else ''}"
                         f"{scen} · {h:.1f} h/day",
                         fontsize=10.5, color=INK, loc="left",
                         fontweight="bold" if c == 0 else "normal")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 14))
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.015)
    cb.set_label("direct sun · hours/day (June-August)", fontsize=10,
                 color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK)
    cb.outline.set_visible(False)
    fig.suptitle(f"Before / after at the proposal points — {CFG['label']}",
                 fontsize=14, x=0.075, y=0.97, ha="left", fontweight="bold",
                 color=INK)
    fig.savefig(f"{OUT_DIR}/proposal_points.png", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_DIR}/proposal_points.png")


if __name__ == "__main__":
    main()
