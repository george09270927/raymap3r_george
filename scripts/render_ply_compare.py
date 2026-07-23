"""Headless side-by-side render of fused point clouds from ONE fixed viewpoint (HANDOFF §7).

Renders each input .ply as a matplotlib 3D scatter with shared axis limits and a shared
camera (elev/azim), so Mode 1 / Mode 2 / Mode 3 comparisons are honest.

Usage:
  python scripts/render_ply_compare.py \
      --plys results/run/pointcloud.ply results/run/pointcloud_static.ply \
      --labels "Mode 1: accumulated" "Mode 2: alpha-filtered" \
      --out results/run/compare_m1_m2.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plys", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max_points", type=int, default=150_000, help="Subsample per cloud.")
    p.add_argument("--elev", type=float, default=-150.0,
                   help="Default oblique bird's-eye works for CUT3R worlds (y-down, z-forward).")
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--point_size", type=float, default=0.5)
    p.add_argument("--lim_pct", type=float, default=2.0,
                   help="Axis limits from union percentiles [p, 100-p] (outlier-robust).")
    return p.parse_args()


def load_ply(path, max_points):
    v = PlyData.read(path)["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    cols = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    if len(pts) > max_points:
        sel = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts, cols


def main():
    args = parse_args()
    labels = args.labels or [os.path.basename(p) for p in args.plys]
    assert len(labels) == len(args.plys), "one label per ply"

    clouds = [load_ply(p, args.max_points) for p in args.plys]
    union = np.concatenate([c[0] for c in clouds], 0)
    lo = np.percentile(union, args.lim_pct, axis=0)
    hi = np.percentile(union, 100 - args.lim_pct, axis=0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2  # cubic box -> equal aspect

    fig = plt.figure(figsize=(7 * len(clouds), 7))
    for k, ((pts, cols), lab) in enumerate(zip(clouds, labels)):
        ax = fig.add_subplot(1, len(clouds), k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=args.point_size, linewidths=0)
        ax.view_init(elev=args.elev, azim=args.azim)
        for setl, d in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
            setl(ctr[d] - half, ctr[d] + half)
        ax.set_axis_off()
        ax.set_title(f"{lab}\n({len(pts):,} pts shown)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[render] {args.out}")


if __name__ == "__main__":
    main()
