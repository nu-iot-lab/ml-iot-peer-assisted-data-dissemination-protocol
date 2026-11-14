#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testbed generator with:
- strict 10×10 cm grid (configurable cell size),
- optional connectivity guarantees under a chosen SF and path-loss model,
- gateway selection (random|center|corner),
- optional plotting with device squares + SF rings.

CSV schema: id,x,y,gateway
"""

import argparse
import csv
import math
import os
import random
from typing import List, Tuple, Dict, Set

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
except Exception:
    plt = None

# Typical sensitivities (dBm @ 125 kHz). Adjust if you use other BW/CR/NF.
SF_SENS_DBM = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}


def sf_max_range_m_ref(tx_dbm: float,
                       pl_ref_db: float,
                       d_ref_m: float,
                       gamma: float,
                       sens_dbm: float,
                       margin_db: float = 0.0) -> float:
    """
    Reference-distance log-distance model:
        PL(d) = PL(d_ref) + 10*gamma*log10(d/d_ref)
        RX = TX - PL(d) >= sens + margin
        => d_max = d_ref * 10 ** ((TX - PL(d_ref) - (sens+margin)) / (10*gamma))
    """
    numer = tx_dbm - pl_ref_db - (sens_dbm + margin_db)
    denom = 10.0 * gamma
    if denom <= 0 or d_ref_m <= 0:
        return 0.0
    return d_ref_m * (10.0 ** (numer / denom))


def nearest_node_to_point(points: Dict[int, Tuple[float, float]],
                          target: Tuple[float, float]) -> int:
    tx, ty = target
    best_id = None
    best_d2 = float('inf')
    for nid, (x, y) in points.items():
        d2 = (x - tx) ** 2 + (y - ty) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = nid
    return best_id


def choose_gateways(points: Dict[int, Tuple[float, float]],
                    n_gateways: int,
                    placement: str,
                    rng: random.Random,
                    width_m: float,
                    height_m: float) -> List[int]:
    n_gateways = max(1, int(n_gateways))
    all_ids = list(points.keys())

    if placement == "random":
        return rng.sample(all_ids, k=min(n_gateways, len(all_ids)))

    if placement == "center":
        center_id = nearest_node_to_point(points, (width_m / 2.0, height_m / 2.0))
        if n_gateways == 1:
            return [center_id]
        remaining = [nid for nid in all_ids if nid != center_id]
        extra = rng.sample(remaining, k=min(n_gateways - 1, len(remaining)))
        return [center_id] + extra

    if placement == "corner":
        corner_id = nearest_node_to_point(points, (0.0, 0.0))
        if n_gateways == 1:
            return [corner_id]
        remaining = [nid for nid in all_ids if nid != corner_id]
        extra = rng.sample(remaining, k=min(n_gateways - 1, len(remaining)))
        return [corner_id] + extra

    # Fallback
    return rng.sample(all_ids, k=min(n_gateways, len(all_ids)))


def build_adjacency(points: Dict[int, Tuple[float, float]], r_max: float) -> Dict[int, Set[int]]:
    """Undirected adjacency: edge if Euclidean distance <= r_max."""
    ids = list(points.keys())
    adj: Dict[int, Set[int]] = {i: set() for i in ids}
    for i in range(len(ids)):
        ni = ids[i]
        xi, yi = points[ni]
        for j in range(i + 1, len(ids)):
            nj = ids[j]
            xj, yj = points[nj]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= r_max * r_max:
                adj[ni].add(nj)
                adj[nj].add(ni)
    return adj


def component_sizes(adj: Dict[int, Set[int]]) -> List[int]:
    """Return sizes of all connected components."""
    seen = set()
    sizes = []
    for n in adj:
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        sizes.append(count)
    sizes.sort(reverse=True)
    return sizes


def passes_connectivity(points: Dict[int, Tuple[float, float]],
                        gw_ids: List[int],
                        scope: str,
                        r_max: float) -> Tuple[bool, Dict]:
    """
    Check connectivity constraints.
    scope:
      - 'neighbor': every node has degree >= 1
      - 'gateway': every non-gateway has at least one gateway within r_max
      - 'full': whole graph is one connected component
    Returns (ok, stats)
    """
    adj = build_adjacency(points, r_max)
    stats = {
        "isolated_nodes": [nid for nid, nbrs in adj.items() if len(nbrs) == 0],
        "largest_cc": component_sizes(adj)[0] if len(adj) > 0 else 0,
        "total_nodes": len(points)
    }

    if scope == "neighbor":
        ok = len(stats["isolated_nodes"]) == 0
        return ok, stats

    if scope == "gateway":
        gw_set = set(gw_ids)
        ok = True
        isolated_from_gw = []
        for nid, (x, y) in points.items():
            if nid in gw_set:
                continue
            # any gateway within r_max?
            good = False
            for gid in gw_ids:
                gx, gy = points[gid]
                if (x - gx) ** 2 + (y - gy) ** 2 <= r_max * r_max:
                    good = True
                    break
            if not good:
                ok = False
                isolated_from_gw.append(nid)
        stats["isolated_from_gateway"] = isolated_from_gw
        return ok, stats

    if scope == "full":
        sizes = component_sizes(adj)
        ok = (len(sizes) > 0 and sizes[0] == len(points))
        stats["component_sizes"] = sizes
        return ok, stats

    # Unknown scope -> accept
    return True, stats


def generate_points_on_grid(rng: random.Random,
                            nodes: int,
                            width: float,
                            height: float,
                            cell_size_m: float) -> Dict[int, Tuple[float, float]]:
    """Place nodes on unique grid cells; return positions at cell centers."""
    inv = 1.0 / cell_size_m
    cells_per_meter = int(round(inv))
    cells_x = int(math.floor(width * cells_per_meter))
    cells_y = int(math.floor(height * cells_per_meter))

    occupied: Set[Tuple[int, int]] = set()
    points: Dict[int, Tuple[float, float]] = {}

    for nid in range(nodes):
        tries = 0
        while True:
            ix = rng.randrange(0, cells_x)
            iy = rng.randrange(0, cells_y)
            if (ix, iy) not in occupied:
                occupied.add((ix, iy))
                break
            tries += 1
            if tries > 200000:
                raise RuntimeError("Unexpected saturation while placing nodes; "
                                   "try fewer nodes or a bigger map.")
        x = (ix + 0.5) * cell_size_m
        y = (iy + 0.5) * cell_size_m
        points[nid] = (x, y)
    return points



def _validate_common_args(args, nodes: int) -> int:
    """Validate map/grid settings for a given node count and return cells_per_meter."""
    if nodes <= 0:
        raise SystemExit("node count must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be > 0")
    if args.cell_size_m <= 0:
        raise SystemExit("--cell-size-m must be > 0")
    inv = 1.0 / args.cell_size_m
    if abs(inv - round(inv)) > 1e-9:
        raise SystemExit("--cell-size-m must divide 1.0 exactly (e.g., 0.1, 0.05, 0.2).")

    cells_per_meter = int(round(inv))
    cells_x = int(math.floor(args.width * cells_per_meter))
    cells_y = int(math.floor(args.height * cells_per_meter))
    if nodes > cells_x * cells_y:
        raise SystemExit(f"Too many nodes ({nodes}) for grid capacity ({cells_x * cells_y} cells).")
    return cells_per_meter


def _compute_connectivity_range(args):
    """Return connectivity radius if connectivity checks are enabled, otherwise None."""
    if not args.ensure_connectivity:
        return None
    if args.sf_for_connectivity is None:
        raise SystemExit("--sf-for-connectivity is required when using --ensure-connectivity.")
    sens = SF_SENS_DBM[args.sf_for_connectivity]
    r_conn = sf_max_range_m_ref(
        tx_dbm=args.tx_dbm,
        pl_ref_db=args.pl_ref_db,
        d_ref_m=args.pl_ref_dist_m,
        gamma=args.pl_gamma,
        sens_dbm=sens,
        margin_db=args.ring_margin_db,
    )
    if r_conn <= 0:
        raise SystemExit("Computed connectivity range is non-positive; check inputs.")
    return r_conn


def _generate_one_testbed(args, nodes: int, out_path: str, rng: random.Random, seed_label) -> Tuple[str, List[int], Dict]:
    """Generate one testbed CSV and return (message, gateway_ids, stats)."""
    cells_per_meter = _validate_common_args(args, nodes)
    r_conn = _compute_connectivity_range(args)

    attempt = 0
    while True:
        attempt += 1
        points = generate_points_on_grid(rng, nodes, args.width, args.height, args.cell_size_m)
        gw_ids = choose_gateways(
            points,
            args.gateways,
            args.gateway_placement,
            rng,
            width_m=args.width,
            height_m=args.height,
        )

        ok = True
        stats = {"total_nodes": nodes}
        if args.ensure_connectivity:
            ok, stats = passes_connectivity(points, gw_ids, args.ensure_connectivity, r_conn)

        if ok:
            break
        if attempt >= args.max_attempts:
            print(
                f"Failed to satisfy connectivity ({args.ensure_connectivity}) "
                f"after {attempt} attempts for nodes={nodes}. Last stats: {stats}"
            )
            raise SystemExit(1)

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "gateway"])
        gw_set = set(gw_ids)
        for nid in range(nodes):
            x, y = points[nid]
            w.writerow([nid, f"{x:.3f}", f"{y:.3f}", 1 if nid in gw_set else 0])

    msg = (
        f"Wrote {out_path} with {nodes} nodes; gateways: {sorted(gw_ids)}; "
        f"placement={args.gateway_placement}; area={args.width}x{args.height} m; "
        f"cell={args.cell_size_m} m (capacity per 1 m² = {cells_per_meter**2}); seed={seed_label}."
    )
    if args.ensure_connectivity:
        msg += (
            f" Connectivity: {args.ensure_connectivity} using SF{args.sf_for_connectivity} "
            f"(r≈{r_conn:.1f} m). Stats: {stats}"
        )
    return msg, gw_ids, stats


def _map_label(width: float, height: float) -> str:
    """Compact label for filenames, e.g. 2000x2000 -> 2km."""
    if abs(width - height) < 1e-9:
        if abs(width % 1000.0) < 1e-9:
            return f"{int(width // 1000)}km"
        return f"{int(width)}m"
    return f"{int(width)}x{int(height)}m"


def _run_batch(args):
    if args.nodes_start is None or args.nodes_step is None or args.nodes_end is None:
        raise SystemExit("Batch mode requires --nodes-start, --nodes-step, and --nodes-end.")
    if args.nodes_step <= 0:
        raise SystemExit("--nodes-step must be > 0")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.nodes_start > args.nodes_end:
        raise SystemExit("--nodes-start must be <= --nodes-end")

    os.makedirs(args.out_dir, exist_ok=True)
    master_rng = random.Random(args.seed)
    label = _map_label(args.width, args.height)

    created = []
    for nodes in range(args.nodes_start, args.nodes_end + 1, args.nodes_step):
        for iteration in range(1, args.iterations + 1):
            # Independent deterministic seed for each generated file when --seed is provided.
            file_seed = master_rng.randrange(1 << 31) if args.seed is not None else random.randrange(1 << 31)
            rng = random.Random(file_seed)
            filename = args.batch_name_template.format(
                label=label,
                width=int(args.width),
                height=int(args.height),
                nodes=nodes,
                iteration=iteration,
                seed=file_seed,
            )
            if not filename.endswith(".csv"):
                filename += ".csv"
            out_path = os.path.join(args.out_dir, filename)
            msg, _gw_ids, _stats = _generate_one_testbed(args, nodes, out_path, rng, file_seed)
            created.append(out_path)
            print(msg)

    print(f"Generated {len(created)} testbed files in {args.out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate one or many non-overlapping testbeds on a fixed grid.")

    # Single-file mode
    ap.add_argument("--nodes", type=int, default=None, help="Number of devices to place for single-file mode.")
    ap.add_argument("--out", type=str, default=None, help="Output CSV for single-file mode.")

    # Batch mode
    ap.add_argument("--batch", action="store_true", help="Generate multiple testbeds across node counts and iterations.")
    ap.add_argument("--nodes-start", type=int, default=None, help="First node count for batch mode.")
    ap.add_argument("--nodes-step", type=int, default=None, help="Step between node counts for batch mode.")
    ap.add_argument("--nodes-end", type=int, default=None, help="Final node count for batch mode, inclusive.")
    ap.add_argument("--iterations", type=int, default=1, help="How many independent testbeds to generate per node count.")
    ap.add_argument("--out-dir", type=str, default="testbeds", help="Output directory for batch mode.")
    ap.add_argument(
        "--batch-name-template",
        type=str,
        default="testbed_{label}_{nodes}_{iteration}.csv",
        help=(
            "Filename template for batch mode. Available fields: "
            "{label}, {width}, {height}, {nodes}, {iteration}, {seed}."
        ),
    )

    # Shared map/grid settings
    ap.add_argument("--width", type=float, default=100.0, help="Map width in meters (default 100).")
    ap.add_argument("--height", type=float, default=100.0, help="Map height in meters (default 100).")
    ap.add_argument("--cell-size-m", type=float, default=0.1,
                    help="Grid cell size in meters (must divide 1.0 exactly; default 0.1 = 10 cm).")

    ap.add_argument("--gateways", type=int, default=1, help="How many gateways to flag (default 1).")
    ap.add_argument("--gateway-placement", choices=["random", "center", "corner"], default="random",
                    help="How to pick gateways among placed nodes.")

    ap.add_argument("--seed", type=int, default=None, help="PRNG seed (for reproducibility).")
    ap.add_argument("--plot", action="store_true", help="Visualize the generated map in single-file mode (matplotlib).")

    # Connectivity enforcement
    ap.add_argument("--ensure-connectivity", choices=["neighbor", "gateway", "full"],
                    help="Ensure connectivity under chosen SF/range.")
    ap.add_argument("--sf-for-connectivity", type=int, choices=[7, 8, 9, 10, 11, 12],
                    help="SF to use for connectivity checks (required if --ensure-connectivity).")
    ap.add_argument("--max-attempts", type=int, default=2000,
                    help="Max random layouts to try to satisfy connectivity (default 2000).")

    # SF ring plotting (single-file plot only)
    ap.add_argument("--plot-sf", action="store_true",
                    help="Overlay SF coverage rings (SF7..12 by default) around gateways in single-file plot.")
    ap.add_argument("--sf-list", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12],
                    help="Which SFs to plot (e.g., --sf-list 7 9 12).")

    # Path-loss / radio params (used for connectivity and rings)
    ap.add_argument("--tx-dbm", type=float, default=14.0, help="TX power (dBm).")
    ap.add_argument("--pl-ref-db", type=float, default=76.45,
                    help="Path loss at reference distance d_ref (dB), e.g., 76.45 at 100 m.")
    ap.add_argument("--pl-ref-dist-m", type=float, default=100.0,
                    help="Reference distance d_ref in meters, e.g., 100 or 40.")
    ap.add_argument("--pl-gamma", type=float, default=4.761,
                    help="Path-loss exponent gamma, e.g., 4.761 or 2.08.")
    ap.add_argument("--ring-margin-db", type=float, default=0.0,
                    help="Extra margin (dB) (reduces range).")

    args = ap.parse_args()

    if args.batch:
        if args.plot:
            print("[WARN] --plot is ignored in --batch mode. Use the separate display script for batch maps.")
        _run_batch(args)
        return

    if args.nodes is None:
        raise SystemExit("Single-file mode requires --nodes. For batch mode, use --batch.")
    if args.out is None:
        raise SystemExit("Single-file mode requires --out. For batch mode, use --out-dir.")

    rng = random.Random(args.seed)
    msg, gw_ids, _stats = _generate_one_testbed(args, args.nodes, args.out, rng, args.seed)
    print(msg)

    # Optional plot for single-file mode
    if args.plot:
        if plt is None:
            print("matplotlib not available; cannot plot.")
        else:
            # Re-load generated points from CSV to avoid returning large objects from the generator.
            points = {}
            with open(args.out, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    points[int(row["id"])] = (float(row["x"]), float(row["y"]))
            gw_ids = []
            with open(args.out, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    is_gw = row.get("gateway", row.get("is_gateway", "0")).strip()
                    if is_gw and int(is_gw):
                        gw_ids.append(int(row["id"]))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor("#fafafa")

            half = args.cell_size_m / 2.0
            for nid, (x, y) in points.items():
                rect = Rectangle((x - half, y - half),
                                 width=args.cell_size_m,
                                 height=args.cell_size_m,
                                 linewidth=0.2, edgecolor='k',
                                 facecolor='C0', alpha=0.9, zorder=2)
                ax.add_patch(rect)

            for gid in gw_ids:
                gx, gy = points[gid]
                rect = Rectangle((gx - half, gy - half),
                                 width=args.cell_size_m,
                                 height=args.cell_size_m,
                                 linewidth=1.2, edgecolor='crimson',
                                 facecolor='gold', alpha=1.0, zorder=4)
                ax.add_patch(rect)

            if args.plot_sf:
                radii = {}
                for sf in args.sf_list:
                    sens = SF_SENS_DBM.get(sf)
                    if sens is None:
                        continue
                    r = sf_max_range_m_ref(
                        tx_dbm=args.tx_dbm,
                        pl_ref_db=args.pl_ref_db,
                        d_ref_m=args.pl_ref_dist_m,
                        gamma=args.pl_gamma,
                        sens_dbm=sens,
                        margin_db=args.ring_margin_db,
                    )
                    radii[sf] = r

                print("Computed SF radii (m):", {sf: round(r, 2) for sf, r in radii.items()})
                maxdim = max(args.width, args.height)
                colors = {7: "#1f77b4", 8: "#ff7f0e", 9: "#2ca02c",
                          10: "#d62728", 11: "#9467bd", 12: "#8c564b"}

                for gw in gw_ids:
                    x0, y0 = points[gw]
                    for sf in sorted(radii.keys()):
                        r = radii[sf]
                        if r > 0 and r <= 1.2 * maxdim:
                            circ = Circle((x0, y0), radius=r, fill=False,
                                          linewidth=1.0, alpha=0.35,
                                          edgecolor=colors.get(sf, "gray"),
                                          zorder=1)
                            ax.add_patch(circ)

                legend_lines = [f"SF{sf}: r≈{radii[sf]:.1f} m"
                                for sf in sorted(radii) if radii[sf] > 0]
                if legend_lines:
                    ax.text(0.02, 0.98,
                            "SF rings (ref model):\n" + "\n".join(legend_lines),
                            transform=ax.transAxes, va="top", ha="left",
                            fontsize=9, bbox=dict(boxstyle="round", alpha=0.12), zorder=5)

            xs = [points[n][0] for n in points]
            ys = [points[n][1] for n in points]
            ax.scatter(xs, ys, s=6, c="k", alpha=1.0, zorder=6, label="node centers")
            if gw_ids:
                gx = [points[n][0] for n in gw_ids]
                gy = [points[n][1] for n in gw_ids]
                ax.scatter(gx, gy, s=18, c="red", alpha=1.0, zorder=7, label="gateway centers")

            ax.set_title(f"{args.nodes} nodes on {args.width}×{args.height} m; cell={args.cell_size_m} m")
            ax.set_xlim(0, args.width)
            ax.set_ylim(0, args.height)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

            import matplotlib.lines as mlines
            node_proxy = mlines.Line2D([], [], color='C0', marker='s', linestyle='None',
                                       markersize=6, label='nodes (squares)')
            gw_proxy = mlines.Line2D([], [], color='crimson', marker='s', linestyle='None',
                                     markersize=7, label='gateways (squares)')
            center_proxy = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
                                         markersize=4, label='centers')
            ax.legend(handles=[node_proxy, gw_proxy, center_proxy], loc="upper right")
            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
