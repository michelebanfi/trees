# PoliMi Città Studi — sun exposure model

Quantifies average direct-sun exposure on the ground around the Leonardo campus
(a self-hosted take on what ShadowMap does), to later evaluate where planting
trees would reduce summer sun exposure the most.

## Pipeline

| step | script | output |
|---|---|---|
| 1. fetch OSM buildings/trees/roads (Overpass) | `fetch_osm.py` | `data/osm_cittastudi.json` |
| 2. compute yearly shadow model | `compute_sun_hours.py` | `data/sun_hours.npz` |
| 3. render heatmap | `render_map.py` | `output/sun_map_annual.png` |

Legacy exploration of the municipality tree dataset: `filter_trees.py`,
`make_map.py` (the municipality data is still used — see below).

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

## Validation points

| point | location | annual | June | December |
|---|---|---|---|---|
| P1 45.478528, 9.231236 | street between PoliMi buildings, no shade | 7.8 h/day | 9.9 | 4.6 |
| P2 45.478120, 9.228271 | central area with trees | 6.5 h/day | 8.0 | 5.1 |

P2 is ~2 h/day shadier than P1 in summer (tree canopy) but *sunnier* in
Nov–Jan (bare trees + open space vs. P1's building canyon at low winter sun) —
the pattern good shading design aims for.

## Next steps

- Canopy transparency (a leafed crown transmits ~10–20 % of direct light).
- Irradiance weighting (W/m² by sun elevation) instead of binary sun/shade —
  makes summer-noon shade count more than 8 am sun.
- Building heights from Milano DBT (Database Topografico) instead of defaults.
- "Where to plant" optimizer: greedy search for tree positions maximising
  reduction of summer sun-hours over walkable cells.
