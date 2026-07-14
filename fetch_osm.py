"""Fetch buildings, trees and roads around PoliMi Citta Studi from OpenStreetMap (Overpass API)."""
import json
import urllib.request

# Generous bbox around the Leonardo campus: shadow casters up to ~250 m outside
# the analysis area still matter at low sun elevations.
BBOX = (45.4735, 9.2200, 45.4835, 9.2380)  # (south, west, north, east)
OUT = "data/osm_cittastudi.json"

QUERY = f"""
[out:json][timeout:90];
(
  way["building"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["building"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["natural"="tree"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["natural"="tree_row"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["highway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out body geom;
"""

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

def main():
    data = None
    for url in ENDPOINTS:
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode({"data": QUERY}).encode(),
            headers={"User-Agent": "polimi-sun-exposure-study/0.1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            break
        except Exception as e:
            print(f"{url} failed: {e}")
    if data is None:
        raise SystemExit("all Overpass endpoints failed")

    counts = {}
    for el in data["elements"]:
        tags = el.get("tags", {})
        if "building" in tags:
            key = "buildings"
        elif tags.get("natural") == "tree":
            key = "trees"
        elif tags.get("natural") == "tree_row":
            key = "tree_rows"
        elif "highway" in tags:
            key = "roads"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    print("fetched:", counts)

    import os
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
