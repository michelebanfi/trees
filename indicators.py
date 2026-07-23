"""Quantitative indicators for the shading proposal.

Two parts:
  1. Campus-wide context (baseline vs. a sizeable campus deployment) — kept
     brief, because a localized intervention barely moves campus averages.
  2. The actionable analysis: per-point **cost-benefit curves**. For each
     proposal point A/B/C, both solutions (semi-mature trees and a translucent
     velario canopy) are evaluated at 25% / 50% / 100% coverage of the usable
     plaza around the point, with NO budget cap — so you see, per point, what
     each design costs and how much summer sun / heat it removes.

Metrics per point are taken over the plaza disc (30 m) around the point:
  - direct June-August sun (h/day) and the change vs. baseline;
  - the 12-14 lunch window in sun (min/120);
  - a clear-sky afternoon surface-temperature proxy;
  - % of ground in shade (< 4 h/day summer sun);
  - cost, annual maintenance, and EUR per h/day of summer sun removed.

Run AFTER: compute_sun_hours.py (baseline), scenarios.py + the two campus
re-sims (for the context table), point_studies.py + its 18 re-sims.
Outputs: output/<campus>/indicators.md, proposal_points.png, cost_benefit.png
"""
import json
import math
import os

import numpy as np

from campus_config import CFG, SUN_FILE, FLOWS_FILE, DATA_DIR, OUT_DIR
from compute_sun_hours import to_xy, load_osm, COVER
from render_map import polimi_mask, HEAT_K, microclimate
from route_flows import (load_transit, classify_stops, build_destinations,
                         build_graph, dijkstra, nearest_node)
from scenarios import (dilate, Field, CAMPUS_ZONE_M,
                       TREE_COST, TREE_MAINT, TREE_TIME_Y, TREE_CROWN,
                       SAIL_SIDE, SAIL_COVERAGE,
                       VELARIO_COST_M2, VELARIO_MAINT_M2, VELARIO_TIME_Y,
                       VELARIO_H, VELARIO_TAU)
from point_studies import LEVELS as _LEVELS
from activity_areas import build_masks

SHADE_H = 4.0     # JJA direct sun below this (h/day) counts as "in shade"
T_AIR = 26.0      # typical Milan JJA afternoon air temperature (deg C)
T_COEF = 4.3      # deg C per kWh/m2/day of absorbed direct solar (proxy fit)
LEVELS = [int(round(l * 100)) for l in _LEVELS]   # [25, 50, 100]

CAMPUS = ["baseline", "trees", "sails"]


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

    def absorbed(self):
        """Neighborhood-mixed absorbed solar (kWh/m2/day), cached."""
        if not hasattr(self, "_absorbed"):
            k = np.zeros_like(self.jja_i)
            for cls, kk in HEAT_K.items():
                k[self.cover == cls] = kk
            self._absorbed = microclimate(self.jja_i * k, ~self.bmask,
                                          self.res)
        return self._absorbed


# ---- campus-wide helpers (context only) ------------------------------------
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
    return float((T_AIR + T_COEF * R.absorbed())[zone].mean())


# ---- per-point evaluation ---------------------------------------------------
def footprint_stats(R, xy_list):
    """Summer-sun / heat stats over the ground cells of a footprint."""
    fld = R.fld
    ii, jj = [], []
    for x, y in xy_list:
        i, j = fld.cell(x, y)
        if fld.inside(i, j) and not R.bmask[i, j]:
            ii.append(i); jj.append(j)
    if not ii:
        return None
    sel = (np.array(ii), np.array(jj))
    h = R.jja_h[sel]
    return {"jja_h": float(h.mean()),
            "lunch": float(R.jja_mid[sel].mean()),
            "temp": float((T_AIR + T_COEF * R.absorbed())[sel].mean()),
            "shade_pct": 100.0 * float((h < SHADE_H).mean())}


def rank_areas(base, areas, center):
    """Score each activity polygon by sun / area / centrality -> priority."""
    rows = []
    for k, a in enumerate(areas, 1):
        m = a["mask"]
        sun = float(base.jja_h[m].mean()) if m.any() else 0.0
        dist = math.hypot(a["cxy"][0] - center[0], a["cxy"][1] - center[1])
        rows.append({"id": k, "area": a["area"], "sun": sun, "dist": dist,
                     "cxy": a["cxy"], "xy": a["xy"]})
    smax = max(r["sun"] for r in rows) or 1.0
    amax = max(r["area"] for r in rows) or 1.0
    dmax = max(r["dist"] for r in rows) or 1.0
    for r in rows:
        s, aa = r["sun"] / smax, r["area"] / amax
        c = 1.0 - r["dist"] / dmax
        r["priority"] = (s + aa + c) / 3.0
    rows.sort(key=lambda r: -r["priority"])
    return rows


def point_cost_benefit(base, pt_scn, pt_ras):
    """Assemble the cost-benefit rows per point / solution / coverage level.

    All levels of a point are scored over the SAME region — the full (100%)
    canopy footprint — so the curve shares one denominator."""
    out = {}
    for name in pt_scn:
        eval_xy = pt_scn[name]["sails"][100]["sail_cells"]
        b = footprint_stats(base, eval_xy)
        plaza_area = pt_scn[name]["sails"][100]["canopy_area_m2"]
        entry = {"base": b, "plaza_area": plaza_area, "sails": [], "trees": []}
        for sol in ("sails", "trees"):
            for lvl in LEVELS:
                sc = pt_scn[name][sol][lvl]
                m = footprint_stats(pt_ras[name][sol][lvl], eval_xy)
                if sol == "trees":
                    qty_n = sc["n_trees"]
                    qty = f"{qty_n} trees"
                    cost = qty_n * TREE_COST
                    maint = qty_n * TREE_MAINT
                else:
                    area = sc["canopy_area_m2"]
                    qty = f"{area:,.0f} m2"
                    cost = area * VELARIO_COST_M2
                    maint = area * VELARIO_MAINT_M2
                dsun = b["jja_h"] - m["jja_h"]
                dtemp = b["temp"] - m["temp"]
                eff = cost / dsun if dsun > 0.05 else None
                entry[sol].append({"lvl": lvl, "qty": qty, "cost": cost,
                                   "maint": maint, "m": m, "dsun": dsun,
                                   "dtemp": dtemp, "eff": eff})
        out[name] = entry
    return out


def main():
    rasters = {s: Raster(npz_path(s)) for s in CAMPUS}
    base = rasters["baseline"]
    d0 = np.load(npz_path("baseline"), allow_pickle=False)
    buildings, _, _, _ = load_osm()
    pmask = polimi_mask(d0, buildings, tuple(base.extent))
    zone = dilate(pmask, CAMPUS_ZONE_M) & ~base.bmask

    scen_pts = {"trees": json.load(open(f"{DATA_DIR}/scenario_trees.json")),
                "sails": json.load(open(f"{DATA_DIR}/scenario_sails.json"))}
    n_trees_new = len(scen_pts["trees"]["trees"])
    n_sails_new = len(scen_pts["sails"]["sails"])

    # ---- per-point cost-benefit scenarios --------------------------------
    pt_scn, pt_ras = {}, {}
    for name in CFG["proposal_points"]:
        if not os.path.exists(f"{DATA_DIR}/scenario_pt{name}_sails_100.json"):
            continue                          # no drawn footprint for this point
        pt_scn[name], pt_ras[name] = {}, {}
        for sol in ("trees", "sails"):
            pt_scn[name][sol], pt_ras[name][sol] = {}, {}
            for lvl in LEVELS:
                pt_scn[name][sol][lvl] = json.load(open(
                    f"{DATA_DIR}/scenario_pt{name}_{sol}_{lvl}.json"))
                pt_ras[name][sol][lvl] = Raster(
                    npz_path(f"pt{name}_{sol}_{lvl}"))
    cb = point_cost_benefit(base, pt_scn, pt_ras)

    # ---- campus-wide context --------------------------------------------
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
    green_m2 = float((base.cover[zone] == COVER["grass"]).sum()) * base.res ** 2
    canopy = {"baseline": 0.0,
              "trees": n_trees_new * math.pi * (TREE_CROWN / 2) ** 2,
              "sails": n_sails_new * SAIL_SIDE ** 2 * SAIL_COVERAGE}
    ntrees = {"baseline": trees_base, "trees": trees_base + n_trees_new,
              "sails": trees_base}

    # ---- reduced 'activity area' context + per-polygon ranking -----------
    areas, union, center = build_masks(base.fld, base.bmask)
    area_shade = {s: 100.0 * float((R.jja_h[union] < SHADE_H).mean())
                  for s, R in rasters.items()}
    area_h = {s: float(R.jja_h[union].mean()) for s, R in rasters.items()}
    area_temp = {s: float((T_AIR + T_COEF * R.absorbed())[union].mean())
                 for s, R in rasters.items()}
    union_m2 = float(union.sum()) * base.res ** 2
    ranking = rank_areas(base, areas, center)

    # ---- report ----------------------------------------------------------
    L = []
    L.append(f"# Shading proposal indicators — {CFG['label']}\n")
    L.append("Two solutions are compared: **semi-mature trees** and a "
             "**translucent velario** (a large tensile shade canopy strung "
             f"over the plaza at ~{VELARIO_H:.0f} m, "
             f"{VELARIO_TAU * 100:.0f}% net direct-sun transmission — larger "
             "and more transparent than discrete shade sails). Direct sun is "
             "the June-August daily mean; a cell counts as *in shade* below "
             f"{SHADE_H:.0f} h/day. Surface temperature is a clear-sky "
             f"afternoon proxy (T = {T_AIR:.0f} degC + {T_COEF} x absorbed "
             "direct solar, mixed over ~10 m). Economic figures and their "
             "sources are in the **Cost basis** section at the end.\n")

    # per-point cost-benefit (the focus)
    L.append("## Per-point cost-benefit (no budget cap)\n")
    L.append("Shading is placed as explicit rectangular structures over each "
             "plaza (one horizontal canopy at A and B; a series of rectangles "
             "at C, from the drawn placements.geojson), with trees planted in "
             "TWO rows along the long edges leaving a clear path down the "
             "middle. Both solutions are sized to 25 / 50 / 100% of the "
             "structure (coverage grows along the corridor from the centre); "
             "all metrics are averaged over the ground of the full (100%) "
             f"canopy footprint. Trees take ~{TREE_TIME_Y:.1f} y to establish; "
             f"the velario ~{VELARIO_TIME_Y:.0f} y (design, permits, install).\n")

    def cb_table(rows, qty_label, base_row):
        t = [f"| coverage | {qty_label} | cost (EUR) | maint (EUR/y) | "
             "JJA sun (h/day) | change | lunch (min/120) | in shade | "
             "surf. temp (degC) | change | EUR per h/day removed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
        b = base_row
        t.append(f"| baseline | - | - | - | {b['jja_h']:.1f} | - | "
                 f"{b['lunch']:.0f} | {b['shade_pct']:.0f}% | {b['temp']:.1f} "
                 "| - | - |")
        for r in rows:
            m = r["m"]
            eff = f"{r['eff']:,.0f}" if r["eff"] else "-"
            t.append(f"| {r['lvl']}% | {r['qty']} | {r['cost']:,.0f} | "
                     f"{r['maint']:,.0f} | {m['jja_h']:.1f} | "
                     f"-{r['dsun']:.1f} | {m['lunch']:.0f} | "
                     f"{m['shade_pct']:.0f}% | {m['temp']:.1f} | "
                     f"-{r['dtemp']:.1f} | {eff} |")
        return t

    for name in cb:
        la, lo = CFG["proposal_points"][name]
        e = cb[name]
        b = e["base"]
        L.append(f"### Point {name} ({la:.6f}, {lo:.6f}) — full canopy "
                 f"{e['plaza_area']:,.0f} m2 (100%), baseline JJA sun "
                 f"{b['jja_h']:.1f} h/day\n")
        L.append("**Velario** (translucent HDPE canopy)\n")
        L += cb_table(e["sails"], "canopy", b)
        L.append("")
        L.append("**Trees** (semi-mature deciduous, planted)\n")
        L += cb_table(e["trees"], "count", b)
        L.append("")

    ratios = [e["sails"][-1]["eff"] / e["trees"][-1]["eff"]
              for e in cb.values()
              if e["sails"][-1]["eff"] and e["trees"][-1]["eff"]]
    ratio = sum(ratios) / len(ratios) if ratios else 0
    gp = max(cb, key=lambda n: cb[n]["sails"][-1]["dsun"]
             - cb[n]["trees"][-1]["dsun"])
    vg, tg = cb[gp]["sails"][-1]["dsun"], cb[gp]["trees"][-1]["dsun"]
    L.append(f"**Reading the curves.** Per h/day of summer sun removed, trees "
             f"are ~{ratio:.0f}x cheaper than the velario. But the trees are "
             "planted as two rows along the edges with an open path down the "
             "middle, so they shade the sides while the central path stays "
             "sunny — at full build-out they therefore remove noticeably less "
             f"total sun than the velario, which covers the whole footprint "
             f"(at point {gp}: trees -{tg:.1f} vs velario -{vg:.1f} h/day). "
             "It is a genuine tradeoff: trees are far cheaper, green the "
             "corridor and keep it walkable, but leave the path exposed; the "
             "velario is the premium option that shades the entire area — path "
             "included — in full from day one, and works over hardscape where "
             "planting is impossible. The practical answer is often a mix: tree "
             "rows along the edges plus a velario over the central path.\n")

    # campus context
    L.append("## Campus-wide context\n")
    L.append("A localized intervention barely shifts campus-wide averages — "
             "which is why the actionable analysis is per-point above. For "
             "reference, an earlier campus deployment "
             f"({n_trees_new} trees / {n_sails_new} legacy 8x8 sail modules) "
             "changes the campus figures as follows (physical outcomes only; "
             "cost is meaningful only at the point scale):\n")
    L.append("| Indicator | Baseline | Trees | Sails |")
    L.append("|---|---|---|---|")
    L.append("| % inter-building routes in shade (JJA) | "
             + " | ".join(f"{shade_routes[s]:.1f}%" for s in CAMPUS) + " |")
    L.append("| Mean sun on inter-building routes (h/day) | "
             + " | ".join(f"{route_h[s]:.2f}" for s in CAMPUS) + " |")
    L.append("| Flow-weighted sun on arrival routes (h/day) | "
             + " | ".join(f"{arrivals_h[s]:.2f}" for s in CAMPUS) + " |")
    L.append("| % campus area under shade (JJA) | "
             + " | ".join(f"{shade_area[s]:.1f}%" for s in CAMPUS) + " |")
    L.append("| Mean sun over campus area (h/day) | "
             + " | ".join(f"{zone_h[s]:.2f}" for s in CAMPUS) + " |")
    L.append("| Avg surface temperature (degC) | "
             + " | ".join(f"{temp[s]:.1f}" for s in CAMPUS) + " |")
    L.append("| Trees on campus | "
             + " | ".join(f"{ntrees[s]:,}" for s in CAMPUS) + " |")
    L.append("| Green area (m2) | " + " | ".join(
        f"{green_m2:,.0f}" for _ in CAMPUS) + " |")
    L.append("| New shade canopy (m2) | " + " | ".join(
        f"{canopy[s]:,.0f}" for s in CAMPUS) + " |")
    L.append("")

    # reduced activity-area context
    L.append("### Restricted to the walkable / activity areas\n")
    L.append(f"The whole-campus averages above dilute the interventions across "
             f"a large mostly-empty surface. Restricting the same indicators to "
             f"the {len(areas)} mapped walkable / dwell polygons "
             f"(**{union_m2:,.0f} m2** where people actually are) gives a "
             "sharper read:\n")
    L.append("| Indicator | Baseline | Trees | Sails |")
    L.append("|---|---|---|---|")
    L.append("| % activity area under shade (JJA) | "
             + " | ".join(f"{area_shade[s]:.1f}%" for s in CAMPUS) + " |")
    L.append("| Mean sun over activity area (h/day) | "
             + " | ".join(f"{area_h[s]:.2f}" for s in CAMPUS) + " |")
    L.append("| Avg surface temperature (degC) | "
             + " | ".join(f"{area_temp[s]:.1f}" for s in CAMPUS) + " |")
    L.append("")

    # per-polygon priority ranking
    L.append("## Which activity areas matter most\n")
    L.append("Each mapped polygon is scored on three axes to steer where to "
             "look next: **sun exposure** (baseline June-August sun, h/day — "
             "higher = more need for shade), **area** (m2 — larger = more people "
             "served), and **centrality** (distance from the activity-area "
             "centre — closer = more used). Priority is the equal-weight mean of "
             "the three, each normalised 0-1 across polygons (centrality = "
             "1 - dist/max). Sorted most-important first; IDs match "
             "`activity_areas.png`.\n")
    L.append("| # | area (m2) | sun exposure (h/day) | dist. from centre (m) | "
             "priority |")
    L.append("|---|---|---|---|---|")
    for r in ranking:
        L.append(f"| P{r['id']} | {r['area']:,.0f} | {r['sun']:.1f} | "
                 f"{r['dist']:.0f} | {r['priority']:.2f} |")
    L.append("")
    top = ", ".join(f"P{r['id']}" for r in ranking[:3])
    L.append(f"Top priorities: **{top}** — large, central and sun-exposed. See "
             "`activity_areas.png` (map + scatter) for the spatial picture.\n")

    L.append(COST_BASIS)

    text = "\n".join(L)
    with open(f"{OUT_DIR}/indicators.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {OUT_DIR}/indicators.md")

    points_figure(base, pt_scn, pt_ras, cb)
    curve_figure(cb)
    activity_figure(base, ranking, center)


COST_BASIS = """## Cost basis (economic estimation — sources)

Planning-grade figures, net of IVA (add ~22% for public tenders) unless noted.

**Trees — semi-mature deciduous (16-18 cm girth), planted**
- Bare supply + planting unit rate **EUR 128.93/tree** (17-18 cm; item
  LOM261.OC.AVA.Pa09...), *Prezzario Regionale dei Lavori Pubblici - Regione
  Lombardia, ed. 2026* — the legal public-tender reference for Lombardy.
  Sensitivity: EUR 113.66 (15-16 cm) ... EUR 151.32 (19-20 cm). Excludes
  staking, de-paving, IVA.
- **Value used: EUR 500/tree** — the realistic all-in on a *paved* urban site
  (~EUR 250 on already-green ground), bundling design, de-paving, planting and
  ~3-5 y establishment care. Source: **Forestami** program figures (reported
  2020-21, corroborated across sources). The point plazas are paved, so EUR 500
  is the applicable figure.
- Maintenance **EUR 100/tree/y** in the establishment years (Forestami);
  establishment ~3 y (up to 5 y maintained); plant in the dormant season.

**Velario — translucent tensile shade canopy (~6 m over the plaza)**
- **EUR 150/m2 of covered plan area** (central; range EUR 120-200) for an
  **HDPE shade-cloth** canopy on galvanised-steel masts + tensioned cables +
  foundations. HDPE is breathable (trapped hot air escapes) -> better
  open-plaza comfort. A waterproof **PVC-coated membrane** alternative runs
  EUR 250-450/m2. Basis: international budgetary ranges (temembrane.com) +
  Italian fabric rates (Maco Technology); no single Italian public EUR/m2
  benchmark exists -> treat as order-of-magnitude and confirm with 2-3 supplier
  quotes (Metexa, Tensomarket, KE Outdoor Design, Maco).
- Maintenance **EUR 12/m2/y** (~3-5% of capex: seasonal rigging, cleaning,
  fabric amortization; HDPE service life ~8-12 y). Derived — flag as an
  assumption.
- Shade performance: HDPE shade cloth blocks ~70-95% of direct sun (we model
  **28% net transmission**). Timeframe ~4-9 months incl. permitting and
  Soprintendenza clearance (historic campus).

Sources: Prezzario Regionale Lombardia 2026 (regione.lombardia.it); Forestami
(latitudeslife.com, nonsprecare.it, lampoonmagazine.com); temembrane.com;
macotechnology.com; maanta.it / wovar.it (HDPE performance); bdir.com /
sergeferrari.com (membrane); civert.it (install time). All figures are planning
estimates to be confirmed with quotes before tender.
"""


def _sail_mask(R, sc):
    """Reconstruct a velario footprint mask on raster R from its cells."""
    m = np.zeros(R.bmask.shape, bool)
    for x, y in sc.get("sail_cells", []):
        i, j = R.fld.cell(x, y)
        if R.fld.inside(i, j):
            m[i, j] = True
    return m


def points_figure(base, pt_scn, pt_ras, cb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.patches import Circle
    from render_map import SUN_RAMP, INK, BUILDING_FILL

    cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)
    names = list(cb.keys())
    fig, axes = plt.subplots(len(names), 3, figsize=(12.5, 3.9 * len(names)),
                             dpi=170, squeeze=False)
    fig.subplots_adjust(top=0.93, hspace=0.16, wspace=0.06)
    for r, name in enumerate(names):
        la, lo = CFG["proposal_points"][name]
        x, y = to_xy(la, lo)
        fc = np.array(pt_scn[name]["sails"][100]["sail_cells"])
        cx, cy = fc[:, 0].mean(), fc[:, 1].mean()
        Z = max(np.ptp(fc[:, 0]), np.ptp(fc[:, 1])) / 2 + 14
        cols = [("Baseline", base, None, None, cb[name]["base"]["jja_h"]),
                ("Velario 100%", pt_ras[name]["sails"][100],
                 pt_scn[name]["sails"][100], "sails",
                 cb[name]["base"]["jja_h"] - cb[name]["sails"][-1]["dsun"]),
                ("Trees 100%", pt_ras[name]["trees"][100],
                 pt_scn[name]["trees"][100], "trees",
                 cb[name]["base"]["jja_h"] - cb[name]["trees"][-1]["dsun"])]
        for c, (label, R, sc, kind, hval) in enumerate(cols):
            ax = axes[r, c]
            field = np.ma.masked_where(R.bmask, R.jja_h)
            ax.imshow(field, extent=R.extent, cmap=cmap, vmin=0, vmax=14,
                      interpolation="nearest", zorder=1)
            b = np.zeros((*R.bmask.shape, 4))
            b[R.bmask] = to_rgba(BUILDING_FILL)
            ax.imshow(b, extent=R.extent, interpolation="nearest", zorder=2)
            if kind == "sails":
                mask = _sail_mask(R, sc)
                ov = np.zeros((*mask.shape, 4))
                ov[mask] = to_rgba("#37474f", 0.45)
                ax.imshow(ov, extent=R.extent, interpolation="nearest",
                          zorder=3)
            elif kind == "trees":
                for p in sc.get("trees", []):
                    ax.add_patch(Circle((p[0], p[1]), TREE_CROWN / 2,
                                        fill=True, fc="#2e7d32", ec="white",
                                        lw=0.3, alpha=0.8, zorder=4))
            ax.plot(x, y, "x", ms=10, mew=2.6, color=INK, zorder=6)
            ax.set_xlim(cx - Z, cx + Z); ax.set_ylim(cy - Z, cy + Z)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{'Point ' + name + ' - ' if c == 0 else ''}"
                         f"{label} - {hval:.1f} h/day", fontsize=10.5,
                         color=INK, loc="left",
                         fontweight="bold" if c == 0 else "normal")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 14))
    cb_ = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.015)
    cb_.set_label("direct sun - hours/day (June-August)", fontsize=10,
                  color=INK)
    cb_.ax.tick_params(labelsize=9, colors=INK)
    cb_.outline.set_visible(False)
    fig.suptitle(f"Proposal points - full canopy (100%) - {CFG['label']}",
                 fontsize=14, x=0.075, y=0.97, ha="left", fontweight="bold",
                 color=INK)
    fig.savefig(f"{OUT_DIR}/proposal_points.png", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_DIR}/proposal_points.png")


def curve_figure(cb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_map import INK, MUTED

    names = list(cb.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4),
                             dpi=170, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        e = cb[name]
        for sol, color, mk, lab in (("sails", "#37474f", "s", "Velario"),
                                    ("trees", "#2e7d32", "o", "Trees")):
            xs = [r["cost"] / 1000.0 for r in e[sol]]
            ys = [r["dsun"] for r in e[sol]]
            ax.plot(xs, ys, "-", color=color, marker=mk, ms=7, lw=2,
                    label=lab, zorder=3)
            for r, xx, yy in zip(e[sol], xs, ys):
                ax.annotate(f"{r['lvl']}%", (xx, yy), xytext=(4, -9),
                            textcoords="offset points", fontsize=8,
                            color=MUTED)
        ax.set_title(f"Point {name}", fontsize=12, color=INK,
                     loc="left", fontweight="bold")
        ax.set_xlabel("cost (EUR thousand)", fontsize=10, color=INK)
        ax.grid(True, alpha=0.25)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)
    axes[0].set_ylabel("summer sun removed (h/day)", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    fig.suptitle(f"Cost vs. summer sun removed, per point - {CFG['label']}",
                 fontsize=14, x=0.02, y=1.02, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cost_benefit.png", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_DIR}/cost_benefit.png")


def activity_figure(base, ranking, center):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import (LinearSegmentedColormap, to_rgba, Normalize)
    from matplotlib.patches import Polygon as MplPoly
    from render_map import SUN_RAMP, INK, MUTED, BUILDING_FILL

    cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)
    pri = plt.cm.viridis
    pv = [r["priority"] for r in ranking]
    norm = Normalize(min(pv), max(pv))
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    allxy = np.vstack([r["xy"] for r in ranking])
    x0, x1 = allxy[:, 0].min() - 25, allxy[:, 0].max() + 25
    y0, y1 = allxy[:, 1].min() - 25, allxy[:, 1].max() + 25

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=170,
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    field = np.ma.masked_where(base.bmask, base.jja_h)
    axm.imshow(field, extent=base.extent, cmap=cmap, vmin=0, vmax=14,
               interpolation="nearest", alpha=0.5, zorder=1)
    b = np.zeros((*base.bmask.shape, 4))
    b[base.bmask] = to_rgba(BUILDING_FILL)
    axm.imshow(b, extent=base.extent, interpolation="nearest", zorder=2)
    for r in ranking:
        axm.add_patch(MplPoly(r["xy"], closed=True,
                              facecolor=pri(norm(r["priority"])),
                              edgecolor=INK, lw=1.1, alpha=0.72, zorder=3))
        axm.annotate(f"P{r['id']}", r["cxy"], ha="center", va="center",
                     fontsize=8, fontweight="bold", color=INK, zorder=5,
                     path_effects=halo)
    axm.plot(center[0], center[1], "*", ms=20, color="#c62828", mec="white",
             mew=1.3, zorder=6)
    axm.set_xlim(x0, x1); axm.set_ylim(y0, y1); axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    axm.set_title("Activity areas, coloured by priority  (star = campus centre)",
                  fontsize=11.5, color=INK, loc="left", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=pri, norm=norm)
    cbar = fig.colorbar(sm, ax=axm, shrink=0.55, pad=0.01)
    cbar.set_label("priority", fontsize=10, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    cbar.outline.set_visible(False)

    sizes = [40 + 0.055 * r["area"] for r in ranking]
    axs.scatter([r["dist"] for r in ranking], [r["sun"] for r in ranking],
                s=sizes, c=pv, cmap=pri, norm=norm, edgecolor=INK, lw=0.8,
                alpha=0.85, zorder=3)
    for r in ranking:
        axs.annotate(f"P{r['id']}", (r["dist"], r["sun"]), xytext=(5, 4),
                     textcoords="offset points", fontsize=8, color=INK,
                     path_effects=halo)
    axs.set_xlabel("distance from campus centre (m)  —  lower = more central",
                   fontsize=10, color=INK)
    axs.set_ylabel("sun exposure (h/day, June-August)", fontsize=10, color=INK)
    axs.set_title("bubble size = area (m2)", fontsize=11.5, color=INK,
                  loc="left", fontweight="bold")
    axs.grid(True, alpha=0.25)
    for sp in axs.spines.values():
        sp.set_color(MUTED)
    axs.tick_params(colors=INK, labelsize=9)

    fig.suptitle(f"Activity-area priorities — {CFG['label']}", fontsize=14,
                 x=0.02, y=1.0, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/activity_areas.png", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_DIR}/activity_areas.png")


if __name__ == "__main__":
    main()
