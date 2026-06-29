#!/usr/bin/env python3
"""
RayMap3R — self-contained streaming inference entry point.

Runs the RayMap3R dual-branch streaming reconstruction model on an image
sequence (a directory of frames, or a video file) and writes per-frame depth,
confidence, color, camera pose/intrinsics, the camera trajectory, and a fused
world point cloud (.ply) to an output directory. Headless — no viewer / no
viser dependency.

The model math lives untouched under ./src (dust3r + croco). This file only
prepares inputs, drives the recurrent rollout, and serializes results.

The adaptive rotation router runs by default (RayMap3R paper, arXiv 2603.20588): it picks the
per-sequence update rule and state-aware smoothing automatically, so no flags are required for
the default setting.

Example:
  python infer.py \
      --frames_dir /path/to/frames \
      --weights weights/cut3r_512_dpt_4_64.pth \
      --output_dir /path/to/result \
      --size 512
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse
from copy import deepcopy

import numpy as np
import torch

# Resolve imports regardless of cwd. The package mixes conventions:
#   - most files use `from dust3r...` / `from croco...`  (needs <root>/src on path)
#   - some files use `from src.dust3r...`                 (needs <root> on path)
# Put both on sys.path up front so import order never matters (the original demo
# only worked because model.py self-appends src/ as a side effect on import).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="RayMap3R self-contained streaming inference.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames_dir", type=str, default=None,
                     help="Directory of image frames (sorted lexicographically).")
    src.add_argument("--video", type=str, default=None,
                     help="Video file; frames are extracted with cv2.")
    p.add_argument("--weights", type=str, default="weights/cut3r_512_dpt_4_64.pth",
                   help="Path to the model checkpoint.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write results.")
    p.add_argument("--size", type=int, default=512,
                   help="Long-side resize for inputs (512 for the dpt model).")
    p.add_argument("--num_frames", type=int, default=30,
                   help="Cap on the number of frames to run (0 = all).")
    p.add_argument("--frame_interval", type=int, default=1,
                   help="Take every k-th frame.")
    p.add_argument("--model_update_type", type=str, default="cut3r",
                   choices=["cut3r", "xattn"],
                   help="cut3r = no gating baseline; xattn = attention gate + RayMap3R alpha gate.")
    p.add_argument("--reset_interval", type=int, default=50,
                   help="Memory reset period in frames (paper default = 50; reset metric "
                        "alignment is applied at each boundary). Set huge to disable.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conf_thr", type=float, default=3.0,
                   help="Confidence threshold for points written to the fused .ply.")
    p.add_argument("--max_ply_points", type=int, default=2_000_000,
                   help="Random-subsample the fused point cloud to at most this many points.")
    return p.parse_args()


def collect_frames(args):
    """Return a sorted list of image paths, extracting from video if needed."""
    if args.frames_dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.frames_dir, "*"))
                       if os.path.splitext(p.lower())[1] in IMG_EXTS)
        if not paths:
            raise SystemExit(f"No images found in {args.frames_dir}")
        if args.frame_interval > 1:
            paths = paths[::args.frame_interval]
        if args.num_frames > 0:
            paths = paths[:args.num_frames]
        return paths, None

    # Video path: extract frames into a temp dir under output_dir/_frames.
    import cv2
    out_frames = os.path.join(args.output_dir, "_frames")
    os.makedirs(out_frames, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(range(0, total, args.frame_interval))
    if args.num_frames > 0:
        idxs = idxs[:args.num_frames]
    paths = []
    for j, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        fp = os.path.join(out_frames, f"{j:06d}.png")
        cv2.imwrite(fp, frame)
        paths.append(fp)
    cap.release()
    if not paths:
        raise SystemExit(f"No frames extracted from {args.video}")
    return paths, out_frames


def build_views(img_paths, size, reset_interval):
    """Image-only view dicts for the recurrent rollout (mirrors demo.prepare_input)."""
    from src.dust3r.utils.image import load_images
    images = load_images(img_paths, size=size)
    views = []
    for i in range(len(images)):
        img = images[i]["img"]
        views.append({
            "img": img,
            "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        })
    return views


def save_results(outputs, outdir, conf_thr, max_ply_points):
    """Serialize per-frame depth/conf/color/camera + trajectory + fused point cloud."""
    import imageio.v2 as iio
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    preds = outputs["pred"]
    views = outputs["views"]
    n = len(preds)

    pts_self = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)  # (N,H,W,3)
    conf_self = torch.cat([p["conf_self"].cpu() for p in preds], 0)          # (N,H,W)
    poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds]  # list (1,4,4)

    B, H, W, _ = pts_self.shape
    pp = torch.tensor([W // 2, H // 2]).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts_self, pp, focal_mode="weiszfeld")

    for sub in ("depth", "conf", "color", "camera"):
        d = os.path.join(outdir, sub)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    poses_c2w = np.stack([poses[i][0].numpy() for i in range(n)], 0)  # (N,4,4)
    all_pts, all_cols = [], []
    for i in range(n):
        depth = pts_self[i, ..., 2].numpy()
        conf = conf_self[i].numpy()
        color = (0.5 * (views[i]["img"].permute(0, 2, 3, 1).cpu() + 1.0))[0].numpy()
        c2w = poses_c2w[i]
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = float(focal[i])
        K[0, 2], K[1, 2] = float(pp[i, 0]), float(pp[i, 1])
        np.save(os.path.join(outdir, "depth", f"{i:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{i:06d}.npy"), conf)
        iio.imwrite(os.path.join(outdir, "color", f"{i:06d}.png"), (color * 255).astype(np.uint8))
        np.savez(os.path.join(outdir, "camera", f"{i:06d}.npz"), pose=c2w, intrinsics=K)

        world = geotrf(poses[i], pts_self[i].unsqueeze(0))[0].numpy().reshape(-1, 3)
        m = (conf.reshape(-1) > conf_thr)
        all_pts.append(world[m])
        all_cols.append((color.reshape(-1, 3)[m] * 255).astype(np.uint8))

    np.save(os.path.join(outdir, "poses_c2w.npy"), poses_c2w)
    # TUM-style trajectory: frame tx ty tz (rotation kept in poses_c2w.npy / camera/*.npz)
    with open(os.path.join(outdir, "trajectory.txt"), "w") as f:
        for i in range(n):
            t = poses_c2w[i][:3, 3]
            f.write(f"{i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

    pts = np.concatenate(all_pts, 0) if all_pts else np.zeros((0, 3), np.float32)
    cols = np.concatenate(all_cols, 0) if all_cols else np.zeros((0, 3), np.uint8)
    if len(pts) > max_ply_points:
        sel = np.random.choice(len(pts), max_ply_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    write_ply(os.path.join(outdir, "pointcloud.ply"), pts, cols)

    return {"num_frames": n, "height": H, "width": W,
            "mean_conf": float(conf_self.mean()), "ply_points": int(len(pts))}


def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        buf = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf["x"], buf["y"], buf["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        buf["r"], buf["g"], buf["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        f.write(buf.tobytes())


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    import torch as _t
    if device == "cuda" and not _t.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        device = "cpu"

    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    img_paths, _tmp = collect_frames(args)
    print(f"[info] {len(img_paths)} frames | size={args.size} | update={args.model_update_type}")

    views = build_views(img_paths, args.size, args.reset_interval)

    print(f"[info] loading model from {args.weights}")
    model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        outputs, _state = inference_recurrent_lighter(views, model, device)
    dt = time.time() - t0
    fps = len(views) / dt if dt > 0 else float("nan")
    print(f"[info] inference done: {dt:.2f}s, {fps:.2f} FPS")

    summary = save_results(outputs, args.output_dir, args.conf_thr, args.max_ply_points)
    summary.update({
        "frames": len(img_paths), "seconds": round(dt, 2), "fps": round(fps, 2),
        "model_update_type": args.model_update_type,
        "weights": os.path.abspath(args.weights),
        "source": args.frames_dir or args.video,
    })
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[info] results ->", args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
