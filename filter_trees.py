"""Filter the Milano municipality tree census (ds2484, whole city) down to the
current campus fetch bbox."""
import json
from collections import Counter

from campus_config import FETCH_BBOX, MUNI_FILE

SRC = "ds2484_alberi_20240331.geojson"

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

S, W, N, E = FETCH_BBOX
kept = []
for ft in data["features"]:
    coords = ft.get("geometry", {}).get("coordinates")
    if not coords or len(coords) < 2:
        continue
    lon, lat = coords[0], coords[1]
    if S <= lat <= N and W <= lon <= E:
        kept.append(ft)

print(f"total features: {len(data['features'])}")
print(f"within {FETCH_BBOX}: {len(kept)}")
genera = Counter((ft["properties"].get("genere") or "Sconosciuto")
                 for ft in kept)
print(f"unique genera: {len(genera)}; top:",
      ", ".join(f"{g} {c}" for g, c in genera.most_common(8)))

out = {
    "type": "FeatureCollection",
    "name": "alberi_campus",
    "crs": data.get("crs"),
    "features": kept,
}
with open(MUNI_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("wrote", MUNI_FILE)
