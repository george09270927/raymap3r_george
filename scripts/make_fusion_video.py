"""Temporal-accumulation fusion video: Mode 1 vs Mode 2 side by side (HANDOFF §7).

Rebuilds per-frame world points offline from a finished infer.py run
(depth/*.npy + camera/*.npz + color/*.png + alpha/*.npy) and renders the fused
cloud growing frame by frame from ONE fixed viewpoint:
  left  = Mode 1 accumulated (confidence filter only)  -> ghost trail forms
  right = Mode 2 alpha-filtered (conf AND alpha >= thr) -> stays clean

Warm-up frames without alpha contribute confidence-filtered points to BOTH panels.

Usage:
  python scripts/make_fusion_video.py --run_dir results/lady_full_force_auto [--alpha_thr 0.5]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as iio
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--alpha_thr", type=float, default=0.5)
    p.add_argument("--conf_thr", type=float, default=3.0,
                   help="Same default as infer.py --conf_thr.")
    p.add_argument("--pts_per_frame", type=int, default=5000,
                   help="Random subsample of each frame's contribution (render speed).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--elev", type=float, default=-150.0)
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--lim_pct", type=float, default=2.0)
    p.add_argument("--out", type=str, default=None,
                   help="Default <run_dir>/fusion_m1_vs_m2.mp4")
    return p.parse_args()


def unproject(depth, K, c2w):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def main():
    args = parse_args()
    rd = args.run_dir
    out = args.out or os.path.join(rd, "fusion_m1_vs_m2.mp4")
    depth_paths = sorted(glob.glob(os.path.join(rd, "depth", "*.npy")))
    if not depth_paths:
        raise SystemExit(f"No depth maps in {rd}/depth")

    rng = np.random.default_rng(0)
    per_frame = []  # (pts, cols, static_mask) already subsampled
    for dp in depth_paths:
        idx = os.path.splitext(os.path.basename(dp))[0]
        depth = np.load(dp)
        conf = np.load(os.path.join(rd, "conf", f"{idx}.npy"))
        cam = np.load(os.path.join(rd, "camera", f"{idx}.npz"))
        color = iio.imread(os.path.join(rd, "color", f"{idx}.png")).reshape(-1, 3) / 255.0
        world = unproject(depth, cam["intrinsics"], cam["pose"])

        m = conf.reshape(-1) > args.conf_thr
        ap = os.path.join(rd, "alpha", f"{idx}.npy")
        if os.path.exists(ap):
            ms = m & (np.load(ap).reshape(-1) >= args.alpha_thr)
        else:
            ms = m  # warm-up: no alpha yet
        keep = np.flatnonzero(m)
        if len(keep) > args.pts_per_frame:
            keep = rng.choice(keep, args.pts_per_frame, replace=False)
        per_frame.append((world[keep].astype(np.float32),
                          color[keep].astype(np.float32),
                          ms[keep]))

    union = np.concatenate([f[0] for f in per_frame], 0)
    lo = np.percentile(union, args.lim_pct, axis=0)
    hi = np.percentile(union, 100 - args.lim_pct, axis=0)
    ctr, half = (lo + hi) / 2, (hi - lo).max() / 2

    writer = iio.get_writer(out, fps=args.fps, codec="libx264", quality=7)
    acc_p, acc_c, acc_s = [], [], []
    for t, (pts, cols, s) in enumerate(per_frame):
        acc_p.append(pts); acc_c.append(cols); acc_s.append(s)
        P = np.concatenate(acc_p, 0); C = np.concatenate(acc_c, 0)
        S = np.concatenate(acc_s, 0)

        fig = plt.figure(figsize=(14, 7))
        for k, (pp, cc, title) in enumerate((
                (P, C, f"Mode 1: accumulated  (frame {t}, {len(P):,} pts)"),
                (P[S], C[S], f"Mode 2: alpha>={args.alpha_thr}  ({int(S.sum()):,} pts)"))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c=cc, s=0.5, linewidths=0)
            ax.view_init(elev=args.elev, azim=args.azim)
            for f, d in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
                f(ctr[d] - half, ctr[d] + half)
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
        plt.close(fig)
    writer.close()
    print(f"[fusion-video] {len(per_frame)} frames -> {out}")


if __name__ == "__main__":
    main()
