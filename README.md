# PoliMi campuses — sun exposure model

Quantifies average direct-sun exposure on the ground around the PoliMi
campuses (a self-hosted take on what ShadowMap does), to later evaluate where
planting trees would reduce summer sun exposure the most.

## Two campuses

Every script is campus-parameterized via the `CAMPUS` env var
(`campus_config.py` holds bboxes, projection origin, POIs, DBT tiles):

    CAMPUS=leonardo python3 fetch_osm.py    # Città Studi (default)
    CAMPUS=bovisa   python3 fetch_osm.py    # Bovisa (both building clusters)

Data lands in `data/<campus>/`, figures in `output/<campus>/`. The Leonardo
analysis box covers campus **plus every walking corridor carrying ≥ 1 % of PT
arrivals** (Lambrate FS included); Bovisa covers the La Masa and Durando
clusters plus 450 m. Campus buildings are drawn light-blue on all maps
(non-PoliMi buildings gray — the heuristic deliberately excludes the
Università Statale buildings that share Città Studi).

## Pipeline (per campus)

| step | script | output in `data|output/<campus>/` |
|---|---|---|
| 1. fetch OSM buildings/trees/roads (Overpass) | `fetch_osm.py` | `osm_campus.json` |
| 2. filter municipality tree census to bbox | `filter_trees.py` | `alberi_filtered.geojson` |
| 3. fetch OSM ground cover | `fetch_landcover.py` | `osm_landcover.json` |
| 4. extract DBT heights/trees from DWG tiles | `dbt_extract.py` | `dbt_extract.json` |
| 5. compute yearly shadow model | `compute_sun_hours.py` | `sun_hours.npz` |
| 6. render heatmaps | `render_map.py` | `sun_map_*.png`, `heat_map_summer.png` |
| 7. fetch PT stops + walk network (larger bbox) | `fetch_transit.py` | `osm_transit.json` |
| 8. pedestrian flows PT → campus + sun overlay | `route_flows.py` | `route_flows.json` |
| 9. render flow map + corridor ranking | `render_flows.py` | `flow_map_summer.png`, `flow_corridors.png` |

| 10. campus shading placements (trees / sails) | `scenarios.py` | `scenario_trees.json`, `scenario_sails.json`, `proposal_placements.png` |
| 11. campus re-simulation | `SCENARIO=data/<c>/scenario_trees.json python3 compute_sun_hours.py` (× trees, sails) | `sun_hours_<scenario>.npz` |
| 12. per-point cost-benefit scenarios | `point_studies.py` | `scenario_pt{A,B,C}_{trees,sails}_{25,50,100}.json` |
| 13. per-point re-simulations (windowed, fast) | `SCENARIO=data/<c>/scenario_pt*.json python3 compute_sun_hours.py` (× 18) | `sun_hours_pt*.npz` |
| 14. indicators + point views + curves + activity-area ranking | `indicators.py` (reads `areas.geojson`) | `indicators.md`, `proposal_points.png`, `cost_benefit.png`, `activity_areas.png` |

`make_map.py` is a legacy exploration of the municipality tree dataset.

## Shading proposal (steps 10–14)

Two solutions are compared: semi-mature **trees** and a **velario** (a large
translucent tensile shade canopy strung over a whole plaza — the reference
typology, larger and more transparent than discrete shade sails).

**Cost basis (real sources, see `indicators.md`):** trees **EUR 500/tree**
all-in on a paved urban site (Forestami program figures; the Prezzario
Regionale Lombardia 2026 bare supply+plant rate is EUR 129/tree), EUR 100/tree/y
maintenance; **velario EUR 150/m²** of covered plan area (HDPE shade-cloth on
steel masts, central of a EUR 120–200 range; PVC membrane alternative
EUR 250–450/m²), EUR 12/m²/y maintenance. All figures are cited in the report's
Cost-basis section and are planning-grade (confirm with quotes before tender).

**Velario model:** a flat translucent canopy at ~6 m with **28 % net
direct-sun transmission** (gaps + HDPE fabric). In `compute_sun_hours.py` it is
a dedicated sheet layer (`sail_cells` + `sail_h` + `sail_tau` in the scenario
JSON): a ground cell is attenuated when its sun ray crosses the canopy plane
inside the footprint (offset computed directly, so a thin sheet is never
over-stepped at high sun — unlike the old crown-blob sails).

**Campus context (steps 10–11):** a greedy optimizer places trees / legacy 8×8
sail modules where their June–August shadow removes the most irradiance from
flow-weighted walked cells (sun field damped after each placement so overlaps
don't double-count). Kept only as context — a localized intervention barely
moves campus-wide averages. Scenarios are *seeded* (`SEED_N` objects forced
within 30 m of each point before the rest is optimized globally).

**Activity areas (`activity_areas.py`, reported by `indicators.py`):** because
the whole-campus averages dilute small interventions, the campus indicators are
*also* computed over the union of hand-mapped walkable / dwell polygons
(`data/<campus>/areas.geojson`, ~59,000 m² — the global figures are kept too).
Each polygon is then scored on **sun exposure** (baseline JJA h/day), **area**,
and **centrality**, into an equal-weight **priority** (0–1), to point future
work at the areas that matter most. Centrality is computed **two ways**: (1)
distance from the geometric activity centroid, and (2) distance from the
**pedestrian-path centroid** — the flow-weighted centre of mass of the
PT→campus walking paths (`route_flows`), **clipped to a campus-scale box**
(`data/<campus>/flow_clip.geojson`) so the long approach corridors to the
stations don't drag it off-campus. This rewards areas on the main internal
circulation; with the clip it lands ~75 m from the geometric centre (so the two
rankings largely agree — a useful cross-check).
Output: two ranking tables in `indicators.md` plus `activity_areas.png`
(geometric) and `activity_areas_flow.png` (path-based, with the pedestrian
network overlaid) — each a priority-coloured map + a sun-vs-centrality bubble
chart (bubble size = area).

**Per-point cost-benefit (steps 12–14, the focus):** `point_studies.py` reads
hand-drawn shading footprints from `data/<campus>/placements.geojson` (one
polygon/loop per point, mapped to the nearest A/B/C — A and B are horizontal
corridor rectangles, C an irregular N–S corridor loop). The **velario** fills
the footprint; **trees** are planted in **two rows along the long edges,
leaving a clear path down the middle**. With **no budget cap** it builds both at
**25 / 50 / 100 %** of the structure (coverage grows along the corridor from the
centre), and all metrics are scored over the ground of the full footprint. Each
scenario carries a rectangular receptor `window` (`[lat, lon, half_x_m,
half_y_m]`) so `compute_sun_hours.py` re-simulates only a small box around the
structure (obstacles stay full-grid) — seconds per run. `indicators.py` turns the 18 re-sims into per-point
cost-benefit tables (cost, maintenance, JJA sun removed, lunch window, %
in shade, surface-temperature proxy, EUR per h/day removed) and the
`cost_benefit.png` curves. Headline: trees remove summer sun ~9× cheaper per
h/day than the velario; the velario's value is shading hard paving without
consuming ground and giving full shade from day one.

## Method

- **Obstacles** are rasterised at 1.5 m into a 2.5D height field:
  buildings as opaque prisms, tree crowns as cylinders from crown base
  (`max(2 m, 0.35·h)`) to top — sun can pass under a crown near the trunk.
- **Sun positions** from the NOAA solar algorithm (checked against known Milan
  values, ~0.2° accurate): the 15th of every month, 15-minute steps, positions
  below 3° elevation ignored.
- **Shadow test**: for every ground cell, ray-march toward the sun over the
  height field (up to 330 m); the cell is shaded if any obstacle exceeds the
  ray height. Metric = **hours of direct sun per day**, per month and annual mean.
- **Seasonality**: deciduous trees cast no shade Nov–Mar (bare canopy);
  `leaf_cycle=evergreen` trees always do.

## Data & defaults (the weak points to improve)

- Ground cover: OSM landuse/leisure polygons + buffered OSM roads, then
  **overridden by the DBT 2D surveyed surface polygons** (`dbt_cover_extract.py`
  chains the CAD boundary fragments of `B102_PEDONALE`, `G401_AIUOLA_*`,
  sport/water/parking layers into rings; road areas never close within their
  layer, so asphalt keeps coming from OSM buffering). Surface heat uses a
  ~10 m normalized-Gaussian neighbourhood mixing (`microclimate` in
  `render_map.py`) so a path between lawns reads cooler than one in asphalt.
- Building heights: only ~29 % of local buildings have OSM `height`/`building:levels`.
  Fallbacks: `levels × 3.2 m + 1` (4.2 m/level for university/hospital/church),
  else per-type defaults (`BUILDING_DEFAULT_H` in `compute_sun_hours.py`).
- Trees: OSM has positions but almost no sizes. Each OSM tree is matched to the
  nearest Milano municipality tree (≤ 8 m) to inherit measured height + crown
  diameter; unmatched municipality trees are added as well (union of sources);
  `natural=tree_row` ways get a stem every 4 m. Remaining unknowns use the local
  medians (7 m height, 4 m crown).
- Flat terrain; trees fully opaque in leaf; direct sun only (no diffuse light).

## Validation points (model v2: DBT heights + semi-transparent crowns)

Leonardo:

| point | location | annual | JJA | DJF |
|---|---|---|---|---|
| P1 45.478528, 9.231236 | street between PoliMi buildings, no shade | 8.5 h/day | 10.4 | 6.1 |
| P2 45.478120, 9.228271 | central area with trees | 6.0 h/day | 7.0 | 5.0 |

Bovisa (open ex-industrial fabric — much sunnier than Città Studi):

| point | location | annual | JJA | DJF |
|---|---|---|---|---|
| B1 45.504546, 9.157647 | La Masa cluster | 11.2 h/day | 14.1 | 8.1 |
| B2 45.504890, 9.163209 | Durando cluster | 9.7 h/day | 11.4 | 7.8 |

P2 stays ~3.4 h/day shadier than P1 in summer (tree canopy) while the winter
gap shrinks to ~1 h (bare crowns transmit 55 %) — summer shade at a modest
winter cost. Both Bovisa points sit in full sun through the entire 12–14
lunch window in summer.

## Pedestrian flows (steps 4–6)

Which walking routes PT users take to campus, and how sunny they are:

- **Origins**: every PT access point near campus — metro *street entrances*
  (Piola M2, Lambrate M2), Lambrate FS, tram/bus stop pairs clustered by name
  (≤ 650 m from the campus-buildings rectangle; a stop is dropped when a
  same-mode stop within 300 m sits ≥ 50 m closer — riders stay on board to
  the closest stop of their line; covered ways such as station underpasses
  count as fully shaded). Weights = assumed modal split
  (`MODAL_SPLIT` in `route_flows.py`: metro .55 / train .20 / tram .15 /
  bus .10 — no public per-stop ATM ridership exists), split evenly within a
  mode.
- **Destinations**: the OSM-mapped campus buildings *inside the analysis
  extent only* (Città Studi; PoliMi sites elsewhere are out of scope), entered
  at tagged `entrance` nodes where available, weighted by footprint × floors.
- **Assignment**: shortest walking path (Dijkstra, steps ×1.4) per
  origin–destination pair on the OSM walk network; flows accumulate per
  segment. Each used segment then samples the June–August sun-hours /
  insolation rasters (`bmask` cells skipped).
- **Metric**: `person_exposure = flow × length × summer insolation` — the
  corridors whose shading would spare people the most summer sun.
- **Headline comparison**: the flow-weighted summer sun on walked meters is
  **5.8 h/day at Leonardo vs 8.9 h/day at Bovisa** (+53 %) — Bovisa's access
  corridors are far more exposed, so the Leonardo methodology pays off even
  more there.

## Next steps

- "Where to plant" optimizer: greedy search for tree positions maximising
  reduction of summer sun-hours, now weightable by pedestrian flow
  (`data/route_flows.json`) so shade lands on the busiest corridors.
- Calibrate `MODAL_SPLIT` / per-station weights (metro share is split evenly
  across stations, which overweights the M4 stations south of Leonardo);
  consider logit assignment over k-shortest paths instead of all-or-nothing
  shortest paths.
- `natural=wood` polygons (e.g. the La Goccia woods at Bovisa) are not
  shadow casters — only point trees are; the wooded area west of La Masa
  therefore reads as full sun.
