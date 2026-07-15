"""Pedestrian flows from public transport to PoliMi Città Studi buildings,
overlaid with the summer sun-exposure raster.

Model: every PT access point (metro entrance, station, tram/bus stop cluster)
is an origin weighted by an assumed modal split; every campus building is a
destination weighted by footprint area x floors (occupancy proxy). Flow of
weight w_origin*w_dest is pushed along the shortest walking path of each
origin-destination pair and accumulated per street segment. Each segment then
samples the sun_hours.npz raster (June-August) to get its summer exposure.

Outputs
  data/route_flows.json   segments with flow + exposure, origins, destinations
  stdout                  ranked "person-sun" corridors
"""
import heapq
import json
import math
from collections import defaultdict

import numpy as np

from campus_config import (CFG, TRANSIT_FILE, OSM_FILE, SUN_FILE,
                           FLOWS_FILE as OUT, is_polimi)
from compute_sun_hours import to_xy, fallback_height, parse_num

# Share of PT arrivals by mode (no public per-stop ATM ridership exists; these
# are assumptions to tune). Within a mode the share is split evenly across its
# access points.
MODAL_SPLIT = {"metro": 0.55, "train": 0.20, "tram": 0.15, "bus": 0.10}

# Stops farther than this (straight line) from the campus-buildings rectangle
# are not plausible arrival stops for the campus.
STOP_CUTOFF_M = 650.0  # keeps Lambrate FS (600 m), drops Argonne/Susa M4 (~720 m)
# A stop is dropped when another same-mode stop within DOMINATE_R m sits at
# least DOMINATE_GAIN m closer to campus — riders stay on board until the
# closest stop of their line.
DOMINATE_R = 300.0
DOMINATE_GAIN = 50.0

# Ways students can walk on. Big roads are included because they have
# sidewalks; motorways/trunks don't exist in the bbox.
WALK_HW = {
    "footway", "path", "pedestrian", "steps", "corridor", "living_street",
    "residential", "service", "unclassified", "tertiary", "secondary",
    "primary", "cycleway", "track", "crossing", "elevator",
}
COST_FACTOR = {"steps": 1.4}  # people avoid stairs a bit

CLUSTER_M = 60.0        # same-name stops within this distance = one stop
ENTRANCE_STATION_M = 150.0  # subway entrance belongs to station within this
SAMPLE_STEP_M = 2.0     # exposure sampling step along segments


# --- geometry helpers --------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ring_area_centroid(xy):
    """Shoelace area (m^2) and centroid of a closed ring [(x, y), ...]."""
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        w = x1 * y2 - x2 * y1
        a += w
        cx += (x1 + x2) * w
        cy += (y1 + y2) * w
    a *= 0.5
    if abs(a) < 1e-9:
        xs, ys = zip(*xy)
        return 0.0, (sum(xs) / len(xs), sum(ys) / len(ys))
    return abs(a), (cx / (6 * a), cy / (6 * a))


# --- transit / origins -------------------------------------------------------
def load_transit():
    data = json.load(open(TRANSIT_FILE))
    nodes, ways, routes = [], [], []
    for el in data["elements"]:
        if el["type"] == "node":
            nodes.append(el)
        elif el["type"] == "way" and "highway" in el.get("tags", {}):
            ways.append(el)
        elif el["type"] == "relation":
            routes.append(el.get("tags", {}))
    return nodes, ways, routes


def classify_stops(nodes):
    """Return stations, subway entrances, tram stops, bus stops, entrances."""
    metro_st, train_st, entr_sub, tram, bus, bldg_entr = [], [], [], [], [], {}
    for n in nodes:
        t = n.get("tags", {})
        xy = to_xy(n["lat"], n["lon"])
        rec = {"id": n["id"], "xy": xy, "lat": n["lat"], "lon": n["lon"],
               "name": t.get("name", "?"), "tags": t}
        rw = t.get("railway")
        if t.get("station") == "subway" or (rw == "station" and t.get("subway") == "yes"):
            metro_st.append(rec)
        elif rw in ("station", "halt"):
            train_st.append(rec)
        elif rw == "subway_entrance":
            entr_sub.append(rec)
        elif rw == "tram_stop":
            tram.append(rec)
        elif t.get("highway") == "bus_stop":
            bus.append(rec)
        elif "entrance" in t:
            bldg_entr[n["id"]] = rec
    return metro_st, train_st, entr_sub, tram, bus, bldg_entr


def cluster_stops(stops):
    """Merge same-name direction pairs into single stop locations."""
    clusters = []
    for s in stops:
        for c in clusters:
            if c["name"] == s["name"] and dist(c["xy"], s["xy"]) < CLUSTER_M:
                c["members"].append(s)
                n = len(c["members"])
                c["xy"] = tuple((c["xy"][i] * (n - 1) + s["xy"][i]) / n for i in (0, 1))
                break
        else:
            clusters.append({"name": s["name"], "xy": s["xy"], "members": [s]})
    return clusters


def rect_dist(xy, extent):
    """Distance from point to the analysis-extent rectangle (0 if inside)."""
    xmin, xmax, ymin, ymax = extent
    dx = max(xmin - xy[0], 0.0, xy[0] - xmax)
    dy = max(ymin - xy[1], 0.0, xy[1] - ymax)
    return math.hypot(dx, dy)


def prune_dominated(stops, rect):
    """Drop stops that have a same-mode neighbor clearly closer to campus."""
    stops = sorted(stops, key=lambda s: rect_dist(s["xy"], rect))
    kept = []
    for s in stops:
        d = rect_dist(s["xy"], rect)
        if any(dist(s["xy"], k["xy"]) < DOMINATE_R
               and rect_dist(k["xy"], rect) < d - DOMINATE_GAIN for k in kept):
            continue
        kept.append(s)
    return kept


def build_origins(metro_st, train_st, entr_sub, tram, bus, extent):
    """PT access points with weights from the modal split."""
    origins = []  # {"xy", "name", "mode", "w"}

    # metro: weight per station, split across its street entrances
    stations = [s for s in metro_st if rect_dist(s["xy"], extent) < STOP_CUTOFF_M]
    stations = prune_dominated(stations, extent)
    if stations:
        w_st = MODAL_SPLIT["metro"] / len(stations)
        for st in stations:
            ent = [e for e in entr_sub if dist(e["xy"], st["xy"]) < ENTRANCE_STATION_M]
            pts = ent if ent else [st]
            for p in pts:
                origins.append({"xy": p["xy"], "name": f"{st['name']} M2",
                                "mode": "metro", "w": w_st / len(pts)})

    # train: station node(s) — Lambrate FS
    tstations = [s for s in train_st if rect_dist(s["xy"], extent) < STOP_CUTOFF_M]
    tstations = prune_dominated(tstations, extent)
    for st in tstations:
        origins.append({"xy": st["xy"], "name": st["name"], "mode": "train",
                        "w": MODAL_SPLIT["train"] / len(tstations)})

    # tram / bus: clustered stops near the campus
    for mode, stops in (("tram", tram), ("bus", bus)):
        cl = [c for c in cluster_stops(stops)
              if rect_dist(c["xy"], extent) < STOP_CUTOFF_M]
        cl = prune_dominated(cl, extent)
        for c in cl:
            origins.append({"xy": c["xy"], "name": c["name"], "mode": mode,
                            "w": MODAL_SPLIT[mode] / len(cl)})
    return origins


# --- destinations ------------------------------------------------------------
def build_destinations(bldg_entrances, extent):
    """Campus buildings inside the analysis extent, weight ~ area x floors."""
    data = json.load(open(OSM_FILE))
    dests = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if "building" not in tags or not is_polimi(tags):
            continue
        if el["type"] == "way" and "geometry" in el:
            ring = [to_xy(g["lat"], g["lon"]) for g in el["geometry"]]
            node_ids = el.get("nodes", [])
        elif el["type"] == "relation":
            outer = [m for m in el.get("members", [])
                     if m.get("role") != "inner" and "geometry" in m]
            if not outer:
                continue
            ring = [to_xy(g["lat"], g["lon"]) for g in outer[0]["geometry"]]
            node_ids = []
        else:
            continue
        area, cen = ring_area_centroid(ring)
        xmin, xmax, ymin, ymax = extent
        if not (xmin <= cen[0] <= xmax and ymin <= cen[1] <= ymax):
            continue  # PoliMi buildings outside the campus area are out of scope
        if any(dist(cen, to_xy(la, lo)) < r
               for la, lo, r in CFG.get("exclude_dest", [])):
            continue  # office-only sites students don't commute to
        h = fallback_height(tags)
        levels = parse_num(tags.get("building:levels")) or max(1, round(h / 4.0))
        doors = [bldg_entrances[i]["xy"] for i in node_ids if i in bldg_entrances]
        dests.append({"name": tags.get("name", f"building {el['id']}"),
                      "xy": cen, "area": area, "levels": levels,
                      "w_raw": area * levels, "doors": doors or [cen]})
    total = sum(d["w_raw"] for d in dests)
    for d in dests:
        d["w"] = d["w_raw"] / total
    return dests


# --- walking graph -----------------------------------------------------------
def build_graph(ways):
    adj = defaultdict(list)   # node id -> [(nbr, cost, length, edge_key)]
    coord = {}                # node id -> (x, y)
    edges = {}                # edge_key -> {"u","v","len","name","highway"}
    for w in ways:
        t = w["tags"]
        if t.get("highway") not in WALK_HW or t.get("foot") == "no" \
                or t.get("access") == "no":
            continue
        f = COST_FACTOR.get(t["highway"], 1.0)
        ids, geo = w.get("nodes", []), w.get("geometry", [])
        if len(ids) != len(geo):
            continue
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i + 1]
            for nid, g in ((u, geo[i]), (v, geo[i + 1])):
                coord.setdefault(nid, to_xy(g["lat"], g["lon"]))
            seg = dist(coord[u], coord[v])
            if seg == 0:
                continue
            key = (min(u, v), max(u, v))
            covered = (t.get("tunnel") not in (None, "no")
                       or t.get("covered") not in (None, "no")
                       or t.get("indoor") == "yes")
            edges[key] = {"u": u, "v": v, "len": seg, "covered": covered,
                          "name": t.get("name", ""), "highway": t["highway"]}
            adj[u].append((v, seg * f, key))
            adj[v].append((u, seg * f, key))

    # keep the largest connected component
    seen, best = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for nbr, _, _ in adj[n]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        if len(comp) > len(best):
            best = comp
    keep = set(best)
    adj = {n: nb for n, nb in adj.items() if n in keep}
    print(f"walk graph: {len(adj)} nodes, {len(edges)} edges "
          f"(largest component kept)")
    return adj, coord, edges


def nearest_node(xy, node_arr, node_ids):
    d2 = (node_arr[:, 0] - xy[0]) ** 2 + (node_arr[:, 1] - xy[1]) ** 2
    return node_ids[int(np.argmin(d2))]


def dijkstra(adj, src):
    distm, prev = {src: 0.0}, {}
    pq = [(0.0, src)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > distm.get(n, math.inf):
            continue
        for nbr, cost, key in adj[n]:
            nd = d + cost
            if nd < distm.get(nbr, math.inf):
                distm[nbr] = nd
                prev[nbr] = (n, key)
                heapq.heappush(pq, (nd, nbr))
    return distm, prev


# --- exposure sampling -------------------------------------------------------
class SunRaster:
    def __init__(self):
        d = np.load(SUN_FILE, allow_pickle=True)
        self.extent = d["extent"]          # xmin xmax ymin ymax (m)
        self.res = float(d["res"][0])
        self.jja_hours = d["hours"][5:8].mean(axis=0)
        self.jja_insol = d["insol"][5:8].mean(axis=0)
        self.bmask = d["bmask"]
        self.ny, self.nx = self.bmask.shape

    def sample(self, xy):
        i = int((self.extent[3] - xy[1]) / self.res)
        j = int((xy[0] - self.extent[0]) / self.res)
        if not (0 <= i < self.ny and 0 <= j < self.nx) or self.bmask[i, j]:
            return None
        return float(self.jja_hours[i, j]), float(self.jja_insol[i, j])


def main():
    nodes, ways, routes = load_transit()
    lines = defaultdict(set)
    for r in routes:
        if r.get("ref"):
            lines[r.get("route")].add(r["ref"])
    print("PT lines in bbox:", {m: sorted(v) for m, v in lines.items()})

    sun = SunRaster()
    extent = tuple(sun.extent)

    metro_st, train_st, entr_sub, tram, bus, bldg_entr = classify_stops(nodes)

    dests = build_destinations(bldg_entr, extent)
    print(f"destinations: {len(dests)} campus buildings, "
          f"total floor area {sum(d['w_raw'] for d in dests)/1e3:.0f} k m2")

    # stop cutoff is measured from the campus buildings, not the (much
    # larger) analysis raster — distant stations are not arrival stops
    dest_rect = (min(d["xy"][0] for d in dests), max(d["xy"][0] for d in dests),
                 min(d["xy"][1] for d in dests), max(d["xy"][1] for d in dests))
    origins = build_origins(metro_st, train_st, entr_sub, tram, bus, dest_rect)
    wsum = sum(o["w"] for o in origins)
    for o in origins:
        o["w"] /= wsum
    by_mode = defaultdict(int)
    for o in origins:
        by_mode[o["mode"]] += 1
    print(f"origins: {len(origins)} access points {dict(by_mode)}")

    adj, coord, edges = build_graph(ways)
    node_ids = list(adj.keys())
    node_arr = np.array([coord[n] for n in node_ids])

    for o in origins:
        o["node"] = nearest_node(o["xy"], node_arr, node_ids)
    for d in dests:
        d["nodes"] = [nearest_node(xy, node_arr, node_ids) for xy in d["doors"]]

    # flow assignment: shortest path per origin x destination door
    flow = defaultdict(float)
    unreached = 0
    for o in origins:
        distm, prev = dijkstra(adj, o["node"])
        for d in dests:
            doors = [n for n in d["nodes"] if n in distm]
            if not doors:
                unreached += 1
                continue
            w_door = o["w"] * d["w"] / len(doors)
            for target in doors:
                n = target
                while n != o["node"]:
                    pn, key = prev[n]
                    flow[key] += w_door
                    n = pn
    if unreached:
        print(f"warning: {unreached} origin-destination pairs unreachable")

    # exposure per used edge
    out_edges = []
    for key, f in flow.items():
        e = edges[key]
        a, b = coord[e["u"]], coord[e["v"]]
        if e["covered"]:  # tunnels/underpasses/arcades get no direct sun
            hours = insol = 0.0
        else:
            n_s = max(2, int(e["len"] / SAMPLE_STEP_M) + 1)
            samples = [sun.sample((a[0] + (b[0] - a[0]) * t,
                                   a[1] + (b[1] - a[1]) * t))
                       for t in np.linspace(0, 1, n_s)]
            samples = [s for s in samples if s is not None]
            hours = float(np.mean([s[0] for s in samples])) if samples else None
            insol = float(np.mean([s[1] for s in samples])) if samples else None
        out_edges.append({
            "a": a, "b": b, "len": e["len"], "name": e["name"],
            "highway": e["highway"], "flow": f,
            "jja_hours": hours, "jja_insol": insol,
            # flow-weighted summer insolation dose picked up on this segment
            "person_exposure": f * e["len"] * insol if insol is not None else None,
        })

    # ranked corridors (aggregate by street name)
    per_street = defaultdict(lambda: [0.0, 0.0, 0.0])  # exp, len, flow*len
    for e in out_edges:
        if e["person_exposure"] and e["name"]:
            s = per_street[e["name"]]
            s[0] += e["person_exposure"]
            s[1] += e["len"]
            s[2] += e["flow"] * e["len"]
    print("\ntop person-sun corridors (flow x summer insolation x length):")
    rank = sorted(per_street.items(), key=lambda kv: -kv[1][0])[:15]
    for name, (exp, ln, fl) in rank:
        print(f"  {name:42s} exposure {exp:8.2f}  "
              f"mean flow {fl/ln:5.1%}  streets len {ln:5.0f} m")

    with open(OUT, "w", encoding="utf-8") as fjson:
        json.dump({
            "edges": out_edges,
            "origins": [{k: o[k] for k in ("xy", "name", "mode", "w")}
                        for o in origins],
            "destinations": [{k: d[k] for k in ("xy", "name", "w")}
                             for d in dests],
            "modal_split": MODAL_SPLIT,
        }, fjson, ensure_ascii=False)
    print(f"\nwrote {OUT} ({len(out_edges)} used segments)")


if __name__ == "__main__":
    main()
