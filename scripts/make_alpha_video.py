"""Overlay the RayMap3R dynamic map (1 - alpha) on RGB frames as a heatmap video.

Reads a finished infer.py output dir (needs color/*.png and alpha/*.npy) and writes
  <run_dir>/alpha_heatmap.mp4         side-by-side [RGB | overlay] video
  <run_dir>/alpha_overlay/*.png       sample overlay frames (every --save_frames_every)

Usage:
  python scripts/make_alpha_video.py --run_dir results/lady_full_auto [--fps 10]
"""
import argparse
import glob
import os

import cv2
import imageio.v2 as iio
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=str, required=True,
                   help="infer.py output dir containing color/ and alpha/.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--blend", type=float, default=0.55,
                   help="Heatmap opacity in the overlay (0..1).")
    p.add_argument("--save_frames_every", type=int, default=10,
                   help="Also dump every k-th overlay frame as png (0 = off).")
    p.add_argument("--normalize", action="store_true",
                   help="Per-frame p5-p95 percentile stretch of (1-alpha) before colormapping. "
                        "Recommended for eyeballing: raw values sit in a narrow high band "
                        "(~0.70-0.78 on lady-running), so relative contrast carries the signal.")
    return p.parse_args()


def main():
    args = parse_args()
    alpha_paths = sorted(glob.glob(os.path.join(args.run_dir, "alpha", "*.npy")))
    if not alpha_paths:
        raise SystemExit(f"No alpha maps in {args.run_dir}/alpha - run infer.py first.")

    suffix = "_norm" if args.normalize else ""
    frame_dir = os.path.join(args.run_dir, f"alpha_overlay{suffix}")
    os.makedirs(frame_dir, exist_ok=True)
    out_mp4 = os.path.join(args.run_dir, f"alpha_heatmap{suffix}.mp4")
    writer = iio.get_writer(out_mp4, fps=args.fps, codec="libx264", quality=7)

    stats = []
    for ap in alpha_paths:
        idx = os.path.splitext(os.path.basename(ap))[0]
        cp = os.path.join(args.run_dir, "color", f"{idx}.png")
        if not os.path.exists(cp):
            continue
        color = iio.imread(cp)                       # (H,W,3) uint8 RGB
        alpha = np.load(ap).astype(np.float32)       # (H,W) staticness in [0,1]
        dyn = np.clip(1.0 - alpha, 0.0, 1.0)         # dynamic-ness
        disp = dyn
        if args.normalize:
            p5, p95 = np.percentile(dyn, 5), np.percentile(dyn, 95)
            disp = np.clip((dyn - p5) / max(p95 - p5, 1e-6), 0.0, 1.0)
        heat = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = ((1 - args.blend) * color + args.blend * heat).astype(np.uint8)

        mean_dyn = float(dyn.mean())
        stats.append((idx, mean_dyn))
        cv2.putText(overlay, f"f{idx}  mean(1-alpha)={mean_dyn:.3f}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.append_data(np.concatenate([color, overlay], axis=1))
        if args.save_frames_every and int(idx) % args.save_frames_every == 0:
            iio.imwrite(os.path.join(frame_dir, f"{idx}.png"),
                        np.concatenate([color, overlay], axis=1))
    writer.close()

    dyns = np.array([s[1] for s in stats])
    print(f"[alpha-video] {len(stats)} frames -> {out_mp4}")
    print(f"[alpha-video] mean(1-alpha): min={dyns.min():.3f} "
          f"median={np.median(dyns):.3f} max={dyns.max():.3f}")
    peak = max(stats, key=lambda s: s[1])
    print(f"[alpha-video] most dynamic frame: {peak[0]} ({peak[1]:.3f})")


if __name__ == "__main__":
    main()
