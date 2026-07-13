import json
import math

SRC = "ds2484_alberi_20240331.geojson"
CENTER_LAT = 45.4772741
CENTER_LON = 9.2285005
RADIUS_M = 1000.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

feats = data["features"]
total = len(feats)
kept = []
for ft in feats:
    coords = ft.get("geometry", {}).get("coordinates")
    if not coords or len(coords) < 2:
        continue
    lon, lat = coords[0], coords[1]
    d = haversine(CENTER_LAT, CENTER_LON, lat, lon)
    if d <= RADIUS_M:
        kept.append(ft)

print(f"total features: {total}")
print(f"within {RADIUS_M:.0f}m: {len(kept)}")

from collections import Counter
genera = Counter()
for ft in kept:
    g = ft["properties"].get("genere") or "Sconosciuto"
    genera[g] += 1

print(f"unique genera: {len(genera)}")
for g, c in genera.most_common():
    print(f"  {g}: {c}")

out = {
    "type": "FeatureCollection",
    "name": "alberi_1km",
    "crs": data.get("crs"),
    "features": kept,
}
with open("alberi_filtered.geojson", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("wrote alberi_filtered.geojson")
