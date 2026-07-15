"""Per-campus configuration for the PoliMi sun-exposure pipeline.

Every script reads its bboxes and file paths from here; select the campus with
the CAMPUS environment variable:

    CAMPUS=leonardo python3 fetch_osm.py   (default)
    CAMPUS=bovisa   python3 fetch_osm.py

Data lands in data/<campus>/, figures in output/<campus>/.
"""
import math
import os

CAMPUS = os.environ.get("CAMPUS", "leonardo")

CONFIGS = {
    "leonardo": dict(
        label="PoliMi Città Studi",
        # projection origin (kept from the original single-campus model)
        origin=(45.4785, 9.2290),
        # analysis (receptor) box: campus + every walking corridor carrying
        # >= 1% of PT arrivals (Lambrate FS included), from route_flows data
        analysis_bbox=(45.4722, 9.2188, 45.4860, 9.2396),  # S, W, N, E
        pois={
            "P1 street (no shade)": (45.478528, 9.231236),
            "P2 central area": (45.478120, 9.228271),
        },
        # 800 m DBT quadrants covering FETCH_BBOX (grid calibrated from the
        # original 6-tile extract; quadrants: 4=NW 1=NE 3=SW 2=SE)
        dbt_tiles=["E09_2", "E10_2", "E10_3", "F09_1", "F09_2", "F10_1",
                   "F10_2", "F10_3", "F10_4", "G09_1", "G10_1", "G10_4"],
        # office-only sites excluded from the flow destinations
        # (lat, lon, radius m): Edificio 32.1-32.5 on via Colombo
        exclude_dest=[(45.471908, 9.227015, 150.0)],
        flow_labels=True,
    ),
    "bovisa": dict(
        label="PoliMi Bovisa",
        origin=(45.5047, 9.1604),
        # covers both building clusters (La Masa side and Durando side)
        # plus 450 m margin
        analysis_bbox=(45.5005, 9.1519, 45.5089, 9.1690),
        pois={
            "B1 La Masa cluster": (45.5045458, 9.1576467),
            "B2 Durando cluster": (45.5048896, 9.1632091),
        },
        dbt_tiles=["D06_1", "D06_2", "D06_3", "D06_4", "D07_1", "D07_2",
                   "D07_3", "D07_4", "E06_1", "E06_4", "E07_1", "E07_4"],
        exclude_dest=[],
        flow_labels=False,
    ),
}

CFG = CONFIGS[CAMPUS]


def pad_bbox(bbox, m):
    s, w, n, e = bbox
    dlat = m / 111132.0
    dlon = m / (111320.0 * math.cos(math.radians((s + n) / 2)))
    return (round(s - dlat, 4), round(w - dlon, 4),
            round(n + dlat, 4), round(e + dlon, 4))


# obstacles up to 250 m outside the analysis box still cast shadows into it
FETCH_BBOX = pad_bbox(CFG["analysis_bbox"], 250)
# PT stops up to ~700 m from the analysis box are plausible arrival stops
TRANSIT_BBOX = pad_bbox(CFG["analysis_bbox"], 800)

DATA_DIR = os.path.join("data", CAMPUS)
OUT_DIR = os.path.join("output", CAMPUS)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

OSM_FILE = os.path.join(DATA_DIR, "osm_campus.json")
COVER_FILE = os.path.join(DATA_DIR, "osm_landcover.json")
TRANSIT_FILE = os.path.join(DATA_DIR, "osm_transit.json")
MUNI_FILE = os.path.join(DATA_DIR, "alberi_filtered.geojson")
DBT_FILE = os.path.join(DATA_DIR, "dbt_extract.json")
SUN_FILE = os.path.join(DATA_DIR, "sun_hours.npz")
FLOWS_FILE = os.path.join(DATA_DIR, "route_flows.json")


def is_polimi(tags):
    """Heuristic: is this OSM building part of the Politecnico campus?"""
    blob = " ".join(str(v) for v in tags.values()).lower()
    return ("politecnico" in blob or "polimi" in blob
            or ("edificio" in blob and any(ch.isdigit() for ch in blob)))
