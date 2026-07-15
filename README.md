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

`make_map.py` is a legacy exploration of the municipality tree dataset.

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
