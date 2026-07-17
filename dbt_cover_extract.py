"""Extract ground-surface polygons from the Milano DBT 2D DWG tiles.

The 2D tiles store surface areas as boundary polylines (CAD style), largely
fragmented. Fragments of one class chain end-to-end into closed rings within
their own layer: we chain them by shared endpoints and keep rings that close.
Road areas never close inside their layer (their boundaries are shared with
other classes), so roads keep coming from OSM buffering in build_cover.

Output: data/<campus>/dbt_cover.json  {class: [[ [E,N], ... ], ...]} in UTM32N
"""
import json
import os
import subprocess
from collections import defaultdict

from campus_config import CFG, DATA_DIR

DWG_DIR = "DWG_2020/2D"
CACHE = os.environ.get("DBT2D_CACHE", "/tmp/dbt2d_json")
TILES = CFG["dbt_tiles"]

# DBT layer -> cover class (paint order is decided in build_cover)
LAYER_CLASS = {
    "B102_PEDONALE": "footpath",
    "C201_ISOLE_PEDONALI": "footpath",
    "B101_FASCIA_SOSTA": "asphalt",
    "B202_TRANVIA": "asphalt",
    "C201_IMPIANTO_SPORT": "sport",
    "C201_FONTANA": "water",
    "C201_PISCINA_SC": "water",
    "G401_AIUOLA_PUBB": "grass",
    "G401_AIUOLA_PRIV": "grass",
}
MAX_RING_M2 = 60_000.0   # reject absurd chained rings
MIN_RING_M2 = 1.0


def shoelace(ring):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def chain_rings(segs):
    """Closed rings from a soup of (partially fragmented) boundary lines."""
    closed = [s[:-1] for s in segs if len(s) >= 4 and s[0] == s[-1]]
    open_ = [s for s in segs if not (len(s) >= 4 and s[0] == s[-1])]
    ends = defaultdict(list)
    for i, s in enumerate(open_):
        ends[s[0]].append(i)
        ends[s[-1]].append(i)
    used = set()
    for start in range(len(open_)):
        if start in used:
            continue
        path = list(open_[start])
        used.add(start)
        while path[0] != path[-1]:
            cands = [i for i in ends[path[-1]] if i not in used]
            if not cands:
                break
            i = cands[0]
            s = open_[i]
            used.add(i)
            path += s[1:] if s[0] == path[-1] else s[::-1][1:]
        if len(path) >= 4 and path[0] == path[-1]:
            closed.append(path[:-1])
    return [r for r in closed
            if MIN_RING_M2 <= shoelace(r) <= MAX_RING_M2]


def parse_tile(path):
    d = json.load(open(path))
    objs = d.get("OBJECTS", d if isinstance(d, list) else [])
    layers = {}
    for o in objs:
        if o.get("object") == "LAYER" and o.get("name"):
            h = o.get("handle")
            layers[h[-1] if isinstance(h, list) else h] = o["name"]
    segs = defaultdict(list)
    for o in objs:
        if o.get("entity") != "LWPOLYLINE":
            continue
        ref = o.get("layer")
        ln = layers.get(ref[2], "?") if isinstance(ref, list) and len(ref) > 2 else "?"
        cls = LAYER_CLASS.get(ln)
        if cls and len(o.get("points", [])) >= 2:
            segs[cls].append([(round(p[0], 2), round(p[1], 2))
                              for p in o["points"]])
    return segs


def main():
    os.makedirs(CACHE, exist_ok=True)
    out = defaultdict(list)
    for tile in TILES:
        dwg = os.path.join(DWG_DIR, f"{tile}.dwg")
        cache = os.path.join(CACHE, f"{tile}.json")
        if not os.path.exists(cache):
            print(f"converting {tile}.dwg ...", flush=True)
            subprocess.run(["dwgread", dwg, "-O", "JSON", "-o", cache],
                           check=True, capture_output=True)
        segs = parse_tile(cache)
        counts = {}
        for cls, ss in segs.items():
            rings = chain_rings(ss)
            out[cls].extend(rings)
            counts[cls] = len(rings)
        print(f"{tile}: " + ", ".join(f"{c} {n}" for c, n in counts.items()),
              flush=True)

    path = os.path.join(DATA_DIR, "dbt_cover.json")
    with open(path, "w") as f:
        json.dump({c: [[list(p) for p in r] for r in rings]
                   for c, rings in out.items()}, f)
    tot = {c: (len(r), round(sum(shoelace(x) for x in r)))
           for c, r in out.items()}
    print("total rings (n, m2):", tot)
    print("wrote", path)


if __name__ == "__main__":
    main()
