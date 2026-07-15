"""Render the pedestrian-flow / summer-sun overlay map and the ranked
person-sun corridor chart from data/route_flows.json."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from campus_config import (CFG, FLOWS_FILE as FLOWS, SUN_FILE as NPZ,
                           TRANSIT_FILE as TRANSIT, OUT_DIR)
from compute_sun_hours import load_osm, to_xy
from render_map import (SUN_RAMP, INK, MUTED, BUILDING_EDGE, halo,
                        draw_context, draw_scalebar)

MODE_MARKER = {"metro": "o", "train": "s", "tram": "^", "bus": "v"}
LW_MIN, LW_MAX = 0.4, 7.0


def lw(flow, fmax):
    return LW_MIN + (LW_MAX - LW_MIN) * math.sqrt(flow / fmax)


def main():
    fl = json.load(open(FLOWS))
    d = np.load(NPZ, allow_pickle=False)
    extent = tuple(d["extent"])
    buildings, _, _, roads = load_osm()
    sun_cmap = LinearSegmentedColormap.from_list("sun", SUN_RAMP)

    edges = fl["edges"]
    fmax = max(e["flow"] for e in edges)
    vmax = math.ceil(max(e["jja_hours"] for e in edges if e["jja_hours"]))

    # context walk network over the full (larger) routing bbox
    ctx = json.load(open(TRANSIT))
    ctx_lines = [np.array([to_xy(g["lat"], g["lon"]) for g in el["geometry"]])
                 for el in ctx["elements"]
                 if el["type"] == "way" and "highway" in el.get("tags", {})]

    # view: analysis extent plus every PT origin
    ox = [o["xy"][0] for o in fl["origins"]]
    oy = [o["xy"][1] for o in fl["origins"]]
    pad = 60
    view = (min(extent[0], min(ox)) - pad, max(extent[1], max(ox)) + pad,
            min(extent[2], min(oy)) - pad, max(extent[3], max(oy)) + pad)

    fig, ax = plt.subplots(figsize=(12.5, 10.5), dpi=200)
    for xy in ctx_lines:
        ax.plot(xy[:, 0], xy[:, 1], color="#e8e6e1", lw=0.6, zorder=1)
    draw_context(ax, d, buildings, roads, extent, show_trees=False)

    # flow segments: width = pedestrian flow, color = summer sun on segment
    order = sorted(edges, key=lambda e: e["flow"])
    segs = [np.array([e["a"], e["b"]]) for e in order]
    cols = [sun_cmap(min(e["jja_hours"], vmax) / vmax) if e["jja_hours"] is not None
            else (0.62, 0.61, 0.59, 1.0) for e in order]
    widths = [lw(e["flow"], fmax) for e in order]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths,
                                     capstyle="round", zorder=6))

    # PT origins: shape = mode, size = share of arrivals
    for o in fl["origins"]:
        ax.plot(*o["xy"], MODE_MARKER[o["mode"]],
                ms=5 + 46 * o["w"], mfc="white", mec=INK, mew=1.5, zorder=7)
    if CFG.get("flow_labels", True):
        labeled, label_pts = {}, []
        for o in sorted(fl["origins"], key=lambda o: -o["w"]):
            name = o["name"]
            if o["w"] < 0.03 or name in labeled or any(
                    math.hypot(o["xy"][0] - p[0], o["xy"][1] - p[1]) < 170
                    for p in label_pts):
                continue
            labeled[name] = True
            label_pts.append(o["xy"])
            ax.annotate(name, o["xy"], xytext=(9, 9),
                        textcoords="offset points", fontsize=9.5, color=INK,
                        fontweight="bold", zorder=8, path_effects=halo())

    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
    draw_scalebar(ax, view)

    sm = plt.cm.ScalarMappable(cmap=sun_cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.015)
    cb.set_label("direct sun on segment · hours per day (June–August)",
                 fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)

    handles = [Line2D([], [], color=INK, lw=lw(f * fmax, fmax),
                      label=f"{f*100:.0f}% of arrivals")
               for f in (0.02, 0.10, 0.40)]
    handles += [Line2D([], [], marker=m, color="none", mfc="white", mec=INK,
                       mew=1.5, ms=9, label=mode)
                for mode, m in MODE_MARKER.items()]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor="none",
                    labelcolor=INK, borderpad=0.9, labelspacing=0.9)
    leg.set_zorder(9)

    ax.set_title("Where students walk, and in how much sun — " + CFG["label"],
                 fontsize=15, color=INK, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.012, "line width = modeled share of PT arrivals walking the segment · "
            "color = direct summer sun along it · gray = outside the shadow model",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(f"{OUT_DIR}/flow_map_summer.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/flow_map_summer.png")

    # --- ranked corridor chart --------------------------------------------
    per_street = {}
    for e in edges:
        if e["person_exposure"] and e["name"]:
            s = per_street.setdefault(e["name"], [0.0, 0.0, []])
            s[0] += e["person_exposure"]
            s[1] += e["len"]
            s[2].append(e["jja_hours"])
    rank = sorted(per_street.items(), key=lambda kv: -kv[1][0])[:12][::-1]
    names = [k for k, _ in rank]
    vals = [v[0] for _, v in rank]
    suns = [np.mean(v[2]) for _, v in rank]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    ax.barh(range(len(rank)), vals, height=0.62, color=SUN_RAMP[3],
            edgecolor="none")
    ax.set_yticks(range(len(rank)))
    ax.set_yticklabels(names, fontsize=10, color=INK)
    for i, (v, s) in enumerate(zip(vals, suns)):
        ax.text(v, i, f"  {s:.1f} h sun", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlabel("person-sun exposure · share of arrivals × summer insolation × length",
                  fontsize=10, color=INK)
    ax.tick_params(labelsize=9, colors=MUTED)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BUILDING_EDGE)
    ax.set_title("Corridors causing the most summer sun exposure",
                 fontsize=13.5, color=INK, loc="left", pad=28, fontweight="bold")
    ax.text(0, 1.015, "named streets only — campus-internal unnamed paths appear "
            "on the map but not here", transform=ax.transAxes, fontsize=8.5,
            color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/flow_corridors.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/flow_corridors.png")


if __name__ == "__main__":
    main()
