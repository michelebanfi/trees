"""Extract building/terrain elevations and trees from Milano DBT 3D DWG tiles.

The DBT (Database Topografico 2020, Comune di Milano) DWG distribution stores,
per 1:1000 map-sheet quadrant:
- POLYLINE_3D on C102_* layers: building outlines with Z = eaves elevation (m a.s.l.)
- INSERT on I201_QPIATTA/QFALDA/QSOM/QCUPOLA/QCURVA: roof elevation points
- INSERT on I201_QPIEDE_EDIFICIO / I201_QTER: building-foot / terrain elevations
- INSERT on G403_ALBERO: surveyed tree positions

Requires `dwgread` (brew install libredwg). Coordinates are UTM 32N (m).
Output: data/dbt_extract.json
"""
import json
import os
import subprocess

TILES = ["F09_1", "F09_2", "F10_1", "F10_2", "F10_3", "F10_4"]
DWG_DIR = "DWG_2020/3D"
CACHE = os.environ.get("DBT_CACHE", "/tmp/dbt_json")
OUT = "data/dbt_extract.json"

ROOF_RING_LAYERS = {
    "C102_EDIFICIO", "C102_EDIF_COMMERC", "C102_EDIF_DI_CULTO",
    "C102_EDIF_INDUSTR", "C102_EDIF_IN_COSTRUZ", "C102_EDIF_MINORE",
    "C102_EDIF_PUBBLICO", "C201_TETTOIA", "C201_BARACCA", "C201_EDIF_MINORE",
}
ROOF_PT_LAYERS = {"I201_QPIATTA", "I201_QFALDA", "I201_QSOM", "I201_QCUPOLA",
                  "I201_QCURVA"}
GROUND_PT_LAYERS = {"I201_QPIEDE_EDIFICIO", "I201_QTER"}
TREE_LAYER = "G403_ALBERO"


def parse_tile(path):
    d = json.load(open(path))
    objs = d["OBJECTS"]

    layers = {}
    for o in objs:
        if o.get("object") == "LAYER":
            layers[o["handle"][2]] = o["name"]

    def lname(o):
        ref = o.get("layer")
        return layers.get(ref[2], "?") if ref else "?"

    rings, roof_pts, ground_pts, trees = [], [], [], []
    cur = None
    for o in objs:
        ent = o.get("entity")
        if ent == "POLYLINE_3D":
            name = lname(o)
            cur = {"layer": name, "pts": []} if name in ROOF_RING_LAYERS else None
        elif ent == "VERTEX_3D" and cur is not None:
            cur["pts"].append(o["point"])
        elif ent == "SEQEND":
            if cur and len(cur["pts"]) >= 3:
                rings.append(cur)
            cur = None
        elif ent == "INSERT":
            name = lname(o)
            p = o.get("ins_pt")
            if not p:
                continue
            if name in ROOF_PT_LAYERS:
                roof_pts.append(p)
            elif name in GROUND_PT_LAYERS:
                ground_pts.append(p)
            elif name == TREE_LAYER:
                trees.append(p[:2])
    return rings, roof_pts, ground_pts, trees


def main():
    os.makedirs(CACHE, exist_ok=True)
    all_rings, all_roof, all_ground, all_trees = [], [], [], []
    for tile in TILES:
        cache = os.path.join(CACHE, f"{tile}.json")
        if not os.path.exists(cache):
            print(f"converting {tile}.dwg ...")
            with open(cache, "w") as f:
                subprocess.run(["dwgread", "-O", "json",
                                os.path.join(DWG_DIR, f"{tile}.dwg")],
                               stdout=f, stderr=subprocess.DEVNULL, check=True)
        rings, roof, ground, trees = parse_tile(cache)
        print(f"{tile}: {len(rings)} roof rings, {len(roof)} roof pts, "
              f"{len(ground)} ground pts, {len(trees)} trees")
        all_rings += rings
        all_roof += roof
        all_ground += ground
        all_trees += trees

    with open(OUT, "w") as f:
        json.dump({"rings": all_rings, "roof_pts": all_roof,
                   "ground_pts": all_ground, "trees": all_trees}, f)
    zs = [p[2] for p in all_ground]
    zs.sort()
    print(f"\ntotal: {len(all_rings)} rings, {len(all_roof)} roof pts, "
          f"{len(all_ground)} ground pts, {len(all_trees)} trees")
    print(f"terrain elevation median {zs[len(zs)//2]:.1f} m a.s.l.")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
