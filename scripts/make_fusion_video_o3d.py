"""Temporal-accumulation fusion video, open3d offscreen edition (HANDOFF §7).

Same story as make_fusion_video.py (left: Mode 1 accumulated, right: Mode 2
alpha-filtered) but rendered with open3d's EGL OffscreenRenderer: full-density
points, perspective camera, camera-trajectory polyline. Headless (no viewer).

Usage (on yoshi):
  python scripts/make_fusion_video_o3d.py --run_dir results/lady_full_force_auto
"""
import argparse
import glob
import os

import cv2
import imageio.v2 as iio
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--alpha_thr", type=float, default=0.5)
    p.add_argument("--conf_thr", type=float, default=3.0)
    p.add_argument("--stride", type=int, default=2,
                   help="Keep every k-th pixel per frame (2 -> ~37k pts/frame @512x288).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--size", type=int, default=720, help="Per-panel render size (square).")
    p.add_argument("--point_size", type=float, default=2.5)
    p.add_argument("--fov", type=float, default=55.0)
    p.add_argument("--eye_offset", type=str, default="0.8,-0.5,-0.8",
                   help="Camera eye = scene_center + offset*extent (x,y,z fractions). "
                        "Default is a right-side 3/4 view chosen on lady-running "
                        "(CUT3R worlds are y-down z-forward).")
    p.add_argument("--camera", type=str, default="fixed", choices=["fixed", "follow"],
                   help="fixed = one viewpoint for the whole video (honest §7 comparisons); "
                        "follow = CUT3R-demo-style flythrough riding behind the estimated camera.")
    p.add_argument("--display", type=str, default="accumulate",
                   choices=["accumulate", "stream"],
                   help="accumulate = naive union of all frames so far (ghost trail forms); "
                        "stream = CUT3R-website style, only the CURRENT frame's points visible "
                        "(their viser playback toggles per-frame node visibility).")
    p.add_argument("--follow_back", type=float, default=0.35,
                   help="follow mode: distance behind the camera (fraction of scene extent).")
    p.add_argument("--follow_up", type=float, default=0.15,
                   help="follow mode: height above the camera (fraction of scene extent).")
    p.add_argument("--follow_smooth", type=float, default=0.25,
                   help="follow mode: EMA weight for the new pose (lower = smoother).")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def unproject(depth, K, c2w, stride):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(0, w, stride, dtype=np.float32),
                       np.arange(0, h, stride, dtype=np.float32))
    z = depth[::stride, ::stride]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def make_pcd(pts, cols):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


def main():
    args = parse_args()
    rd = args.run_dir
    suffix = ("_follow" if args.camera == "follow" else "") + \
             ("_stream" if args.display == "stream" else "")
    out = args.out or os.path.join(rd, f"fusion_m1_vs_m2_o3d{suffix}.mp4")
    depth_paths = sorted(glob.glob(os.path.join(rd, "depth", "*.npy")))
    if not depth_paths:
        raise SystemExit(f"No depth maps in {rd}/depth")
    poses_c2w = np.load(os.path.join(rd, "poses_c2w.npy"))

    per_frame = []
    for dp in depth_paths:
        idx = os.path.splitext(os.path.basename(dp))[0]
        depth = np.load(dp)
        conf = np.load(os.path.join(rd, "conf", f"{idx}.npy"))[::args.stride, ::args.stride]
        cam = np.load(os.path.join(rd, "camera", f"{idx}.npz"))
        color = iio.imread(os.path.join(rd, "color", f"{idx}.png"))[
            ::args.stride, ::args.stride].reshape(-1, 3) / 255.0
        world = unproject(depth, cam["intrinsics"], cam["pose"], args.stride)
        m = conf.reshape(-1) > args.conf_thr
        ap = os.path.join(rd, "alpha", f"{idx}.npy")
        if os.path.exists(ap):
            ms = m & (np.load(ap)[::args.stride, ::args.stride].reshape(-1) >= args.alpha_thr)
        else:
            ms = m
        per_frame.append((world[m].astype(np.float32), color[m].astype(np.float32), ms[m]))

    # fixed viewpoint: above-behind the scene (CUT3R worlds are y-down, z-forward)
    union = np.concatenate([f[0] for f in per_frame], 0)
    lo, hi = np.percentile(union, 2, 0), np.percentile(union, 98, 0)
    center, ext = (lo + hi) / 2, float((hi - lo).max())
    off = np.array([float(x) for x in args.eye_offset.split(",")])
    eye = center + off * ext
    up = np.array([0.0, -1.0, 0.0])

    renderer = rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = args.point_size
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 3.0

    def render_panel(pts, cols, traj, view):
        v_ctr, v_eye, v_up = view
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pc", make_pcd(pts, cols), mat)
        if len(traj) >= 2:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(traj.astype(np.float64))
            ls.lines = o3d.utility.Vector2iVector(
                np.stack([np.arange(len(traj) - 1), np.arange(1, len(traj))], 1))
            ls.paint_uniform_color([1.0, 0.1, 0.1])
            renderer.scene.add_geometry("traj", ls, line_mat)
        renderer.setup_camera(args.fov, v_ctr.astype(np.float32),
                              v_eye.astype(np.float32), v_up.astype(np.float32))
        return np.asarray(renderer.render_to_image())

    # follow mode: ride behind the estimated camera, EMA-smoothed to avoid jitter
    ema_eye, ema_ctr, ema_up = None, None, None

    def follow_view(t):
        nonlocal ema_eye, ema_ctr, ema_up
        c2w = poses_c2w[t]
        pos, fwd, upv = c2w[:3, 3], c2w[:3, 2], -c2w[:3, 1]
        raw_eye = pos - fwd * (args.follow_back * ext) + upv * (args.follow_up * ext)
        raw_ctr = pos + fwd * (0.6 * ext)
        b = args.follow_smooth
        if ema_eye is None:
            ema_eye, ema_ctr, ema_up = raw_eye, raw_ctr, upv
        else:
            ema_eye = (1 - b) * ema_eye + b * raw_eye
            ema_ctr = (1 - b) * ema_ctr + b * raw_ctr
            ema_up = (1 - b) * ema_up + b * upv
        return ema_ctr, ema_eye, ema_up / np.linalg.norm(ema_up)

    writer = iio.get_writer(out, fps=args.fps, codec="libx264", quality=8)
    acc_p, acc_c, acc_s = [], [], []
    for t, (pts, cols, s) in enumerate(per_frame):
        acc_p.append(pts); acc_c.append(cols); acc_s.append(s)
        P = np.concatenate(acc_p, 0); C = np.concatenate(acc_c, 0)
        S = np.concatenate(acc_s, 0)
        traj = poses_c2w[:t + 1, :3, 3]
        view = follow_view(t) if args.camera == "follow" else (center, eye, up)

        if args.display == "stream":
            # left: CUT3R-website style (only the current frame visible);
            # right: what naive persistence of the same stream looks like
            left = render_panel(pts, cols, traj, view)
            right = render_panel(P, C, traj, view)
            l_label = f"Streaming, current frame only (CUT3R-style)  f{t:03d}"
            r_label = f"Naive accumulation  {len(P):,} pts"
        else:
            left = render_panel(P, C, traj, view)
            right = render_panel(P[S], C[S], traj, view)
            l_label = f"Mode 1: accumulated  f{t:03d}  {len(P):,} pts"
            r_label = f"Mode 2: alpha>={args.alpha_thr}  {int(S.sum()):,} pts"
        frame = np.concatenate([left, right], axis=1)
        cv2.putText(frame, l_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, r_label,
                    (args.size + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2,
                    cv2.LINE_AA)
        writer.append_data(frame)
        if t == len(per_frame) - 1:
            iio.imwrite(os.path.splitext(out)[0] + "_final.png", frame)
    writer.close()
    print(f"[fusion-o3d] {len(per_frame)} frames -> {out}")


if __name__ == "__main__":
    main()
