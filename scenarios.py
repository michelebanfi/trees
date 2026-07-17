"""Greedy flow-weighted placement of shading interventions.

Two solutions, sized to the same budget:
  - trees: semi-mature deciduous specimens (Celtis/Platanus class)
  - sails: 6x6 m tensile shade-sail modules at ~4 m

Each candidate ground cell is scored by how much June-August direct
irradiance its shadow would remove from places people are (pedestrian-flow
raster from route_flows + campus open spaces), using a precomputed JJA
"shadow kernel" per object type. Greedy: place, damp the local sun field,
repeat — so overlapping shadows are not double-counted. The exact effect is
then verified by re-running compute_sun_hours with SCENARIO=<json>.

Outputs: data/<campus>/scenario_trees.json, scenario_sails.json,
         output/<campus>/proposal_placements.png
"""
import json
import math

import numpy as np

from campus_config import CFG, SUN_FILE, FLOWS_FILE, DATA_DIR, OUT_DIR
from compute_sun_hours import (RES, LAT0, LON0, load_osm, solar_position,
                               julian_date, clear_sky_direct_horizontal, COVER)

# --- COST ASSUMPTIONS (rough Milan figures — adjust freely) -----------------
BUDGET = 200_000.0          # implementation budget per solution (EUR)

TREE_COST = 1_200.0         # semi-mature tree, planted (pit, stake, 2y care)
TREE_MAINT = 60.0           # EUR/year (watering, pruning)
TREE_TIME_Y = 0.5           # one planting season
TREE_H, TREE_CROWN = 8.0, 6.0   # planted size (m); grows further over ~10 y

# cable-net canopy (Wurzburg-style): light triangular sails strung on
# catenary cables from poles/facades over walking areas, with gaps.
SAIL_SIDE = 8.0             # one module covers an 8x8 m *plan* area
SAIL_COVERAGE = 0.6         # sail fabric covers ~60% of the plan area
SAIL_COST_M2 = 100.0        # per plan m2, hung from existing/light supports
SAIL_COST = SAIL_COST_M2 * SAIL_SIDE ** 2      # 6,400 EUR / module
SAIL_MAINT_M2 = 5.0         # EUR/plan-m2/year (seasonal rigging, cleaning)
SAIL_MAINT = SAIL_MAINT_M2 * SAIL_SIDE ** 2
SAIL_TIME_Y = 1.0           # design + permits + installation
SAIL_H = 7.2                # canopy height (m)
# the fabric (60% of 64 m2) is modeled as one equivalent canopy blob
SAIL_EFF_R = math.sqrt(SAIL_SIDE ** 2 * SAIL_COVERAGE / math.pi)  # ~3.5 m

# --- placement rules ---------------------------------------------------------
CAMPUS_ZONE_M = 60.0        # "campus grounds" = within this of a campus bldg
CORRIDOR_FLOW = 0.01        # corridors: edges carrying >= 1% of arrivals
CORRIDOR_ZONE_M = 12.0
TREE_SPACING = 5.0          # between new trees (m)
SAIL_SPACING = 6.0          # module pitch (m)
TREE_BLDG_CLEAR = 2.5       # trees keep clear of facades (roots/canopy)
KERNEL_MAX_M = 36.0         # ignore shadows cast farther than this
KERNEL_MIN_ELEV = 12.0      # low sun shadows land far away — skip
OPACITY = {"trees": 0.85, "sails": 0.90}
MIN_GAIN_FRAC = 0.05        # stop when benefit < 5% of the first placement

# option A: guarantee visible interventions at the proposal points, then
# spend the remaining budget globally
SEED_RADIUS_M = 30.0
SEED_N = {"trees": 5, "sails": 2}   # forced per proposal point

# option B: independent per-point mini-studies (own budget, own area)
LOCAL_BUDGET = 20_000.0
LOCAL_RADIUS_M = 50.0


def conv_same(a, k):
    """2-D 'same' convolution via FFT (numpy only)."""
    H, W = a.shape
    kh, kw = k.shape
    s = (H + kh - 1, W + kw - 1)
    out = np.fft.irfft2(np.fft.rfft2(a, s) * np.fft.rfft2(k, s), s)
    return out[kh // 2:kh // 2 + H, kw // 2:kw // 2 + W]


def dilate(mask, radius_m):
    r = max(1, int(round(radius_m / RES)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2 <= r * r).astype(np.float32)
    return conv_same(mask.astype(np.float32), disc) > 0.5


class Field:
    """Raster indexing for the sun_hours grid."""
    def __init__(self, d):
        self.extent = d["extent"]
        self.res = float(d["res"][0])
        self.ny, self.nx = d["bmask"].shape

    def cell(self, x, y):
        return (int((self.extent[3] - y) / self.res),
                int((x - self.extent[0]) / self.res))

    def xy(self, i, j):
        return (self.extent[0] + (j + 0.5) * self.res,
                self.extent[3] - (i + 0.5) * self.res)

    def inside(self, i, j):
        return 0 <= i < self.ny and 0 <= j < self.nx


def flow_raster(fld):
    """Pedestrian flow stamped onto grid cells (share of arrivals per cell)."""
    fl = json.load(open(FLOWS_FILE))
    Wf = np.zeros((fld.ny, fld.nx), np.float32)
    for e in fl["edges"]:
        (ax, ay), (bx, by) = e["a"], e["b"]
        n = max(2, int(e["len"] / 1.2) + 1)
        cells = set()
        for t in np.linspace(0, 1, n):
            i, j = fld.cell(ax + (bx - ax) * t, ay + (by - ay) * t)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if fld.inside(i + di, j + dj):
                        cells.add((i + di, j + dj))
        for c in cells:
            Wf[c] += e["flow"]
    return Wf


def shadow_kernel(hc, footprint_r_m, opacity, square):
    """K[di,dj] = fraction of JJA direct irradiance blocked at that offset
    by an object whose canopy center is at height hc."""
    R = int(math.ceil(KERNEL_MAX_M / RES))
    K = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    fr = max(1, int(round(footprint_r_m / RES)))
    yy, xx = np.mgrid[-fr:fr + 1, -fr:fr + 1]
    foot = (np.maximum(abs(yy), abs(xx)) <= fr if square
            else yy ** 2 + xx ** 2 <= fr * fr)
    fidx = np.argwhere(foot) - fr

    minutes = np.arange(3 * 60, 20 * 60, 15).astype(float)
    wsum = 0.0
    for m in (6, 7, 8):
        jd = julian_date(2025, m, 15, minutes)
        elev, az = solar_position(jd, LAT0, LON0)
        for e, a in zip(elev, az):
            if e <= 3.0:
                continue
            I = clear_sky_direct_horizontal(e)
            wsum += I
            if e < KERNEL_MIN_ELEV:
                continue
            L = hc / math.tan(math.radians(e))
            dx = -math.sin(math.radians(a)) * L
            dy = -math.cos(math.radians(a)) * L
            ci = R + int(round(-dy / RES))
            cj = R + int(round(dx / RES))
            ii = fidx[:, 0] + ci
            jj = fidx[:, 1] + cj
            ok = (ii >= 0) & (ii < K.shape[0]) & (jj >= 0) & (jj < K.shape[1])
            K[ii[ok], jj[ok]] += I * opacity
    return K / wsum


def greedy(n_max, cand, W, S, kernel, spacing, allowed=None,
           stop_frac=MIN_GAIN_FRAC):
    """Place up to n_max objects; returns [(i,j)...] and benefit estimate.

    Mutates `cand` (spacing) and `S` (shading) IN PLACE so that sequential
    phases compose (seeds first, then the global remainder). `allowed`
    restricts where this phase may *select* without narrowing the shared
    candidate mask; stop_frac=None disables the diminishing-returns stop
    (used to force seed placements)."""
    K = kernel
    R = K.shape[0] // 2
    Kf = K[::-1, ::-1]
    WS = W * S
    B = conv_same(WS, Kf)
    placed, benefit = [], 0.0
    first = None
    for _ in range(n_max):
        sel = cand if allowed is None else (cand & allowed)
        Bm = np.where(sel, B, -1.0)
        i, j = np.unravel_index(int(np.argmax(Bm)), Bm.shape)
        gain = Bm[i, j]
        if gain <= 0:
            break
        if first is None:
            first = gain
        if stop_frac is not None and gain < first * stop_frac:
            break
        placed.append((i, j))
        benefit += gain
        # damp the sun field under this object's summer shadow
        r0, r1 = max(0, i - R), min(S.shape[0], i + R + 1)
        c0, c1 = max(0, j - R), min(S.shape[1], j + R + 1)
        S[r0:r1, c0:c1] *= (1.0 - K[r0 - i + R:r1 - i + R,
                                    c0 - j + R:c1 - j + R])
        WS[r0:r1, c0:c1] = W[r0:r1, c0:c1] * S[r0:r1, c0:c1]
        # recompute benefit where the change is felt (kernel reach beyond it)
        rr0, rr1 = max(0, r0 - R), min(S.shape[0], r1 + R)
        cc0, cc1 = max(0, c0 - R), min(S.shape[1], c1 + R)
        e0, e1 = max(0, rr0 - R), min(S.shape[0], rr1 + R)
        f0, f1 = max(0, cc0 - R), min(S.shape[1], cc1 + R)
        patch = conv_same(WS[e0:e1, f0:f1], Kf)
        B[rr0:rr1, cc0:cc1] = patch[rr0 - e0:rr1 - e0, cc0 - f0:cc1 - f0]
        # spacing between new objects
        sp = max(1, int(round(spacing / RES)))
        yy, xx = np.mgrid[-sp:sp + 1, -sp:sp + 1]
        disc = yy ** 2 + xx ** 2 <= sp * sp
        g0, g1 = max(0, i - sp), min(S.shape[0], i + sp + 1)
        h0, h1 = max(0, j - sp), min(S.shape[1], j + sp + 1)
        cand[g0:g1, h0:h1] &= ~disc[g0 - i + sp:g1 - i + sp,
                                    h0 - j + sp:h1 - j + sp]
    return placed, benefit


def main():
    d = np.load(SUN_FILE, allow_pickle=False)
    fld = Field(d)
    bmask = d["bmask"]
    cover = d["cover"]
    S = d["insol"][5:8].mean(axis=0)          # JJA kWh/m2/day per cell

    from render_map import polimi_mask
    buildings, _, _, _ = load_osm()
    pmask = polimi_mask(d, buildings, tuple(d["extent"]))

    Wf = flow_raster(fld)
    campus_zone = dilate(pmask, CAMPUS_ZONE_M)
    corridor = dilate(Wf > CORRIDOR_FLOW, CORRIDOR_ZONE_M)
    zone = campus_zone | corridor
    walkable = np.isin(cover, [COVER["paved"], COVER["footpath"],
                               COVER["grass"]]) & ~bmask
    W = Wf / max(Wf.max(), 1e-9) + 0.12 * (campus_zone & walkable)
    W *= ~bmask

    # keep new objects out of existing crowns
    occupied = np.zeros_like(bmask)
    for x, y, r in d["trees"]:
        i, j = fld.cell(x, y)
        rr = max(1, int(round(r / RES)))
        if fld.inside(i, j):
            g0, g1 = max(0, i - rr), min(fld.ny, i + rr + 1)
            h0, h1 = max(0, j - rr), min(fld.nx, j + rr + 1)
            occupied[g0:g1, h0:h1] = True

    near_bldg = dilate(bmask, TREE_BLDG_CLEAR)
    base_cand = zone & ~bmask & ~occupied & (S > 0.3)
    cand_tree = base_cand & ~near_bldg & np.isin(
        cover, [COVER["paved"], COVER["footpath"], COVER["grass"]])
    cand_sail = base_cand & np.isin(cover, [COVER["paved"], COVER["footpath"]])

    n_trees = int(BUDGET // TREE_COST)
    n_sails = int(BUDGET // SAIL_COST)
    specs = {
        "trees": (cand_tree, shadow_kernel(TREE_H - TREE_CROWN * 0.25,
                                           TREE_CROWN / 2, OPACITY["trees"],
                                           square=False),
                  n_trees, TREE_SPACING),
        "sails": (cand_sail, shadow_kernel(SAIL_H, SAIL_EFF_R,
                                           OPACITY["sails"], square=False),
                  n_sails, SAIL_SPACING),
    }

    from compute_sun_hours import to_xy
    pts_xy = {name: to_xy(la, lo)
              for name, (la, lo) in CFG["proposal_points"].items()}
    II, JJ = np.mgrid[0:fld.ny, 0:fld.nx]

    def disc_mask(x, y, r_m):
        i0, j0 = fld.cell(x, y)
        return (II - i0) ** 2 + (JJ - j0) ** 2 <= (r_m / RES) ** 2

    def payload(kind, placed):
        pts = [fld.xy(i, j) for i, j in placed]
        if kind == "trees":
            return {"trees": [[round(x, 1), round(y, 1), TREE_H, TREE_CROWN]
                              for x, y in pts]}
        return {"sails": [[round(x, 1), round(y, 1)] for x, y in pts],
                "sail_h": SAIL_H, "sail_crown": round(2 * SAIL_EFF_R, 1)}

    for kind, (cand, kernel, n_max, spacing) in specs.items():
        unit = TREE_COST if kind == "trees" else SAIL_COST

        # reference: unseeded, purely global optimum (for the sacrifice %)
        S_ref, c_ref = S.copy(), cand.copy()
        _, ref_benefit = greedy(n_max, c_ref, W, S_ref, kernel, spacing)

        # option A: force SEED_N objects near each proposal point, then
        # spend the remaining budget globally
        S_run, c_run = S.copy(), cand.copy()
        placed, benefit, seeded = [], 0.0, {}
        for name, (x, y) in pts_xy.items():
            p, b = greedy(SEED_N[kind], c_run, W, S_run, kernel, spacing,
                          allowed=disc_mask(x, y, SEED_RADIUS_M),
                          stop_frac=None)
            placed += p
            benefit += b
            seeded[name] = len(p)
        p, b = greedy(n_max - len(placed), c_run, W, S_run, kernel, spacing)
        placed += p
        benefit += b

        out = {"name": kind, "budget": BUDGET, "unit_cost": unit,
               "seeded": seeded, "seed_radius_m": SEED_RADIUS_M,
               "benefit": benefit, "benefit_unseeded": ref_benefit}
        out.update(payload(kind, placed))
        path = f"{DATA_DIR}/scenario_{kind}.json"
        json.dump(out, open(path, "w"))
        sac = 100.0 * (1.0 - benefit / ref_benefit)
        print(f"{kind}: {len(placed)}/{n_max} placed "
              f"(EUR {len(placed) * unit:,.0f}), seeded {seeded}, "
              f"benefit {benefit:.1f} vs unseeded {ref_benefit:.1f} "
              f"(sacrifice {sac:.1f}%) -> {path}")

        # option B: independent per-point mini-studies (own budget & area)
        n_local = int(LOCAL_BUDGET // unit)
        for name, (x, y) in pts_xy.items():
            S_l, c_l = S.copy(), cand.copy()
            p, b = greedy(n_local, c_l, W, S_l, kernel, spacing,
                          allowed=disc_mask(x, y, LOCAL_RADIUS_M),
                          stop_frac=None)
            outp = {"name": f"pt{name}_{kind}", "budget": LOCAL_BUDGET,
                    "unit_cost": unit, "radius_m": LOCAL_RADIUS_M,
                    "benefit": b}
            outp.update(payload(kind, p))
            lpath = f"{DATA_DIR}/scenario_pt{name}_{kind}.json"
            json.dump(outp, open(lpath, "w"))
            print(f"  local {name}/{kind}: {len(p)}/{n_local} placed "
                  f"(EUR {len(p) * unit:,.0f}) -> {lpath}")

    render_placements(d, fld, pmask)


def render_placements(d, fld, pmask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, Rectangle
    from render_map import SUN_RAMP, INK, MUTED, BUILDING_FILL, POLIMI_FILL

    extent = tuple(d["extent"])
    jja = d["hours"][5:8].mean(axis=0)
    cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=170)
    for ax, kind, color, label in ((axes[0], "trees", "#2e7d32", "new tree"),
                                   (axes[1], "sails", "#455a64",
                                    f"net module {SAIL_SIDE:.0f}x"
                                    f"{SAIL_SIDE:.0f} m")):
        sc = json.load(open(f"{DATA_DIR}/scenario_{kind}.json"))
        field = np.ma.masked_where(d["bmask"], jja)
        ax.imshow(field, extent=extent, cmap=cmap, vmin=0, vmax=14,
                  interpolation="nearest", alpha=0.55, zorder=1)
        b = np.zeros((*d["bmask"].shape, 4))
        b[d["bmask"]] = matplotlib.colors.to_rgba(BUILDING_FILL)
        b[pmask] = matplotlib.colors.to_rgba(POLIMI_FILL)
        ax.imshow(b, extent=extent, interpolation="nearest", zorder=2)
        pts = sc.get("trees", sc.get("sails", []))
        for p in pts:
            if kind == "trees":
                ax.add_patch(Circle((p[0], p[1]), TREE_CROWN / 2, fill=True,
                                    fc=color, ec="white", lw=0.3, alpha=0.85,
                                    zorder=4))
            else:
                ax.add_patch(Rectangle((p[0] - SAIL_SIDE / 2,
                                        p[1] - SAIL_SIDE / 2),
                                       SAIL_SIDE, SAIL_SIDE, fill=True,
                                       fc=color, ec="white", lw=0.3,
                                       alpha=0.9, zorder=4))
        for name, (la, lo) in CFG["proposal_points"].items():
            from compute_sun_hours import to_xy
            x, y = to_xy(la, lo)
            ax.plot(x, y, "x", ms=9, mew=2.4, color=INK, zorder=6)
            ax.annotate(name, (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=11, color=INK,
                        fontweight="bold", zorder=6)
        ax.set_title(f"{len(pts)} {label}s · EUR {len(pts) * sc['unit_cost']:,.0f}",
                     fontsize=12, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Proposed shading placements — {CFG['label']} "
                 f"(equal budget EUR {BUDGET:,.0f})",
                 fontsize=14, x=0.09, y=0.99, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT_DIR}/proposal_placements.png", bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_DIR}/proposal_placements.png")


if __name__ == "__main__":
    main()
