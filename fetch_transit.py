"""Fetch public-transport stops, station/building entrances and the walking
network around PoliMi Citta Studi from OpenStreetMap (Overpass API).

Larger bbox than fetch_osm.py: students arrive from Lambrate FS / Piola M2 /
tram+bus stops that sit outside the shadow-analysis area, and the walk network
must connect them to the campus.

The download is split into two Overpass requests (PT nodes + route relations,
then the walking network) — the combined query 504s under load.
"""
import json
import time
import urllib.request
import urllib.parse

from campus_config import TRANSIT_BBOX as BBOX, FETCH_BBOX as CAMPUS_BBOX, \
    TRANSIT_FILE as OUT

BB = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
CB = f"{CAMPUS_BBOX[0]},{CAMPUS_BBOX[1]},{CAMPUS_BBOX[2]},{CAMPUS_BBOX[3]}"

QUERY_PT = f"""
[out:json][timeout:180];
(
  node["railway"~"^(station|halt|tram_stop|subway_entrance|train_station_entrance)$"]({BB});
  node["station"="subway"]({BB});
  node["highway"="bus_stop"]({BB});
  node["public_transport"~"^(station|stop_position|platform)$"]({BB});
  node["entrance"]({CB});
);
out body;
relation["route"~"^(bus|tram|subway|train|light_rail|trolleybus)$"]({BB});
out tags;
"""

QUERY_WALK = f"""
[out:json][timeout:180];
way["highway"]({BB});
out body geom;
"""

ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

def fetch(query, tries=2):
    for attempt in range(tries):
        for url in ENDPOINTS:
            req = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": "polimi-sun-exposure-study/0.1"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.load(resp)
            except Exception as e:
                print(f"{url} failed: {e}", flush=True)
        if attempt < tries - 1:
            print("retrying in 30 s ...", flush=True)
            time.sleep(30)
    raise SystemExit("all Overpass endpoints failed")

def main():
    pt = fetch(QUERY_PT)
    print("PT elements:", len(pt["elements"]), flush=True)
    walk = fetch(QUERY_WALK)
    print("walk elements:", len(walk["elements"]), flush=True)

    data = {"elements": pt["elements"] + walk["elements"]}

    counts = {}
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "relation":
            key = f"route_{tags.get('route')}"
        elif "highway" in tags and el["type"] == "way":
            key = "walk_ways"
        elif tags.get("railway") in ("station", "halt") or tags.get("public_transport") == "station":
            key = "stations"
        elif tags.get("railway") in ("subway_entrance", "train_station_entrance"):
            key = "station_entrances"
        elif tags.get("railway") == "tram_stop":
            key = "tram_stops"
        elif tags.get("highway") == "bus_stop":
            key = "bus_stops"
        elif "entrance" in tags:
            key = "building_entrances"
        else:
            key = "other_pt"
        counts[key] = counts.get(key, 0) + 1
    print("fetched:", counts)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
