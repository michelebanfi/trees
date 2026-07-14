"""Fetch ground-cover polygons (grass, water, parking, pitches...) from OSM.

Kept separate from fetch_osm.py: Overpass mirrors often time out on the
combined query, and this layer changes rarely.
"""
import json
import time
import urllib.request
import urllib.parse

BBOX = (45.4735, 9.2200, 45.4835, 9.2380)
OUT = "data/osm_landcover.json"

QUERY = f"""
[out:json][timeout:120];
(
  way["landuse"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["leisure"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["natural"~"water|wood|scrub|grassland"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["amenity"="parking"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
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
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(
                    url, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                    headers={"User-Agent": "polimi-sun-exposure-study/0.1"})
                with urllib.request.urlopen(req, timeout=150) as r:
                    data = json.load(r)
                break
            except Exception as e:
                print(f"{url} failed: {e}")
        if data:
            break
        time.sleep(15)
    if data is None:
        raise SystemExit("all Overpass endpoints failed")
    print("elements:", len(data["elements"]))
    json.dump(data, open(OUT, "w"))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
