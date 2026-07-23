# Run Log

Machine: yoshi (RTX 4090 24GB) unless noted. Env: `conda activate raymap3r`.
All commands run from `~/Warehouse/raymap3r_george` on yoshi.

## 2026-07-22 — Day 1: environment + smoke test + full sample runs

### Environment (scripts/setup_env_yoshi.sh, log: setup_env.log)

Issues hit and fixed along the way (all encoded in the setup script / requirements.txt):

1. Fresh miniconda default channels require interactive ToS acceptance → env created with `-c conda-forge --override-channels`.
2. torch 2.1.x `cpp_extension` needs `pkg_resources.packaging`, removed in setuptools>=70 → pinned `setuptools<70` before building curope.
3. gdown 6.x removed `--fuzzy` → download checkpoint by bare Drive file ID.
4. Latest transformers requires torch>=2.4 and dropped `transformers.file_utils` → pinned `transformers==4.45.2`.
5. `accelerate` imported by model.py/inference.py but missing from upstream requirements → added (1.0.1).
6. CUT3R checkpoint unpickle references `omegaconf` → added.

Sanity: `torch 2.1.1+cu121 | cuda: True | NVIDIA GeForce RTX 4090`; checkpoint `weights/cut3r_512_dpt_4_64.pth` (3.0G, gdown ID `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD`).

### Runs (input: `data/examples/lady-running.mp4` from CUT3R repo, 65 frames, 854x480 → 512x288)

| # | command | output dir | result |
|---|---------|-----------|--------|
| 1 | `python infer.py --video data/examples/lady-running.mp4 --output_dir results/smoke_8f_xattn --num_frames 8 --model_update_type xattn` | `results/smoke_8f_xattn` | 8f, 11.96 FPS, ply 890,516 pts. Output formats validated: depth/conf (288,512) f32, camera npz keys pose+intrinsics, poses (8,4,4), binary ply header OK |
| 2 | `python infer.py --video data/examples/lady-running.mp4 --output_dir results/lady_full_xattn --num_frames 0 --model_update_type xattn` | `results/lady_full_xattn` | 65f, 10.85 FPS, mean_conf 9.8586, ply 2.0M (hit --max_ply_points cap). VRAM sampled mid-run: 8343 MiB @ 81% util |
| 3 | `python infer.py --video data/examples/lady-running.mp4 --output_dir results/lady_full_cut3r --num_frames 0 --model_update_type cut3r` | `results/lady_full_cut3r` | 65f, 11.41 FPS. Outputs bit-identical to run 2 (see finding below) |

### Findings

- **`--model_update_type` has no effect on the recurrent inference path.** Runs 2 and 3
  (xattn vs cut3r) produced bit-identical pose / depth / conf (max abs diff 0.0, measured).
  Cause (read from source): `forward_recurrent_lighter` resets
  `config.model_update_type = "cut3r"` (model.py ~1674), then an adaptive rotation router
  decides the regime once per sequence after a ~20-frame warm-up: median camera rotation
  < 2 deg/frame → `xattn` (gated), else `cut3r` (model.py ~1902-1909).
  Which regime the router picked for lady-running is not instrumented yet.
  → Gate on/off A/B (HANDOFF §3 pose-quality check) requires a small forced-regime patch first.
- **`res["alpha_img"]` is already attached to each frame's pred dict** (model.py ~1888) —
  the Day-2 alpha dump is mostly a `save_results` addition, not inference plumbing.
- Day-1 exit criterion MET: fused `pointcloud.ply` + per-frame `depth/ conf/ color/ camera/`
  + `poses_c2w.npy` + `trajectory.txt` + `summary.json` exist and load correctly.

## 2026-07-22 — Day 2: alpha dump + forced-regime patch + heatmap

### Code changes (commit refs in git; [george] block markers inside inherited files)

- `src/dust3r/model.py`: forced-regime bypass of the rotation router (`model.force_update_type`),
  router decision logging + `last_router_info`; forced `cut3r` also disables `_apply_alpha_gate`
  (a router-decided cut3r regime would still apply it, which is not a clean baseline).
- `infer.py`: `--force_update_type {auto,cut3r,xattn}` (note: old `--model_update_type` is a no-op
  on this path), saves `alpha/{i:06d}.npy` per frame, `summary.json` gains `alpha_frames`,
  `force_update_type`, `router`.
- `scripts/make_alpha_video.py` (new): `[RGB | 1-alpha heatmap]` overlay video + sample pngs,
  `--normalize` = per-frame p5-p95 stretch.

### Runs (lady-running.mp4, 65 frames)

| # | command | output dir | result |
|---|---------|-----------|--------|
| 4 | `python infer.py --video data/examples/lady-running.mp4 --output_dir results/lady_full_force_auto --num_frames 0 --force_update_type auto` | `results/lady_full_force_auto` | router decided **xattn** at frame 20, median_rot **1.79 deg/frame** (threshold 2.0); 60/65 alpha frames (first 5 = warm-up, none); bit-identical to Day-1 runs (patch = zero regression in auto) |
| 5 | same with `--force_update_type xattn` | `results/lady_full_force_xattn` | gated regime from frame 0 |
| 6 | same with `--force_update_type cut3r` | `results/lady_full_force_cut3r` | true vanilla baseline (alpha gate off) |

### Findings

- **A/B switch works**: forced xattn vs forced cut3r differ substantially
  (pose max translation diff 2.36, depth max diff 1.32 at frame 40).
  Day-1 mystery explained: the router picks xattn on lady-running (1.79 < 2.0), so both
  old flag values landed in the same regime.
- **alpha map baseline is high and narrow**: mean(1-alpha) in 0.708-0.782 across frames.
  Absolute values are NOT directly usable as motion probability. With per-frame p5-p95
  normalization the structure is clean: runner + moving people hot, floor/static background cold;
  static furniture edges stay warm (depth-uncertain boundaries).
  -> For Day 4 per-instance aggregation, use RELATIVE scores (per-frame normalized or
  scene-median-relative), not absolute thresholds.
- Videos: `results/lady_full_force_auto/alpha_heatmap.mp4` (raw) and `alpha_heatmap_norm.mp4`
  (normalized; recommended for eyeballing).
- **Day-2 exit criterion MET** (normalized heatmap is a plausible motion map).

## 2026-07-22 — Day 2 (cont.): HANDOFF §7 + Mode-2 static fusion

- HANDOFF gains §7 "Visualization modes" (Mode 1 accumulated / Mode 2 alpha-filtered /
  Mode 3 registry rendering; one fixed viewpoint across modes; headless only).
- `infer.py` now also writes `pointcloud_static.ply`: confidence filter AND
  `alpha >= --alpha_static_thr` (default 0.5; warm-up frames contribute confidence-only).
- `scripts/render_ply_compare.py` (new): fixed-viewpoint headless side-by-side ply renderer
  (default elev=-150 azim=-90 suits CUT3R y-down worlds; viewpoint scanned and chosen on Day 2).

| # | command | output dir | result |
|---|---------|-----------|--------|
| 7 | run 4 rerun (gains static ply, thr 0.5) | `results/lady_full_force_auto` | static 1,086,922 pts vs Mode-1 2.0M cap |
| 8 | same with `--alpha_static_thr 0.25` | `results/lady_m2_thr025` | static hits 2.0M cap |

- Threshold reality check (measured): per-frame MEAN alpha is only ~0.25, but the per-pixel
  distribution is wide, so thr 0.5 is NOT degenerate — it keeps 1.09M points and visually removes
  most of the runner's ghost trail while preserving floor/walls; 0.25 keeps a partial trail.
- Day-2 sanity pair: `results/lady_full_force_auto/compare_m1_m2.png` (Mode 1 vs 0.5 vs 0.25).
- `scripts/make_fusion_video.py` (new): temporal-accumulation video, Mode 1 vs Mode 2 side by
  side from the fixed viewpoint, rebuilt offline from saved depth/camera/color/alpha (no
  re-inference). Output: `results/lady_full_force_auto/fusion_m1_vs_m2.mp4` — ghost trail
  visibly forms on the left while the right stays clean (final frame: 325k vs 47k pts;
  alpha also trims depth-uncertain regions, so Mode 2 is sparser overall, not only de-ghosted).
- `scripts/make_fusion_video_o3d.py` (new, preferred): same story rendered with open3d EGL
  OffscreenRenderer — full-density points (stride 2, ~37k pts/frame, 2.2M final), perspective
  camera, camera-trajectory polyline. Default right-side 3/4 viewpoint chosen from a 4-view scan.
  Output: `results/lady_full_force_auto/fusion_m1_vs_m2_o3d.mp4` (+ `_final.png`).
  Context: CUT3R's pretty demo videos are recordings of their interactive viser WebGL viewer
  showing per-frame streaming (no naive accumulation) — the visual gap vs our first matplotlib
  attempt was rendering tech (sparse scatter), not the reconstruction. open3d + matplotlib added
  to requirements.txt as viz-only deps; open3d 0.19 EGL headless verified on yoshi.

## 2026-07-23 — Why lady-running fusion looks bad: content, not pipeline

Question: our fusion video looks far worse than CUT3R's website demos. Two diagnostics:

| # | command | output dir | result |
|---|---------|-----------|--------|
| 9 | Mode-1 accumulation, first 20 vs all 65 lady frames (inline render) | `/tmp/drift_check.png` | first 20 frames ALREADY equally smeared -> NOT long-horizon pose drift; the smear is ghosting from many moving people + fast tracking camera |
| 10 | `python infer.py --frames_dir data/examples/004 --output_dir results/ex004_auto --num_frames 0 --force_update_type auto` (CUT3R examples/004, 70 frames, gentle camera) | `results/ex004_auto` | router: xattn, median_rot 0.43 deg/frame (vs lady 1.79). Mode-1 accumulation is SMOOTH and CUT3R-demo-like -> pipeline fine, lady-running is just hard content |

- Additional finding from 004: alpha>=0.5 removes 78% of points (2.0M -> 434k). Correction
  after the follow-cam render: 004 is NOT mostly static — a person walks toward the camera
  (gentle camera + single mover). The filter removes her ghost trail correctly BUT also eats
  much of the static furniture/floor — the absolute threshold over-filters statics.
  Reinforces: per-instance aggregation (Day 4) must use RELATIVE alpha statistics.
- `make_fusion_video_o3d.py --camera follow` (new): CUT3R-demo-style flythrough riding behind
  the estimated camera (EMA-smoothed). On 004 this view shows a textbook ghost trail in Mode 1
  against a crisp static background -> 004 promoted to the showcase clip; outputs
  `results/{ex004_auto,lady_full_force_auto}/fusion_m1_vs_m2_o3d_follow.mp4`.
- CONFIRMED how CUT3R's website "Dynamic Scene Reconstruction" videos work
  (CUT3R repo `viser_utils.py:735-736`): per-frame point clouds are separate viser nodes and
  playback toggles node VISIBILITY — only the current frame is ever shown. No accumulation,
  hence no ghosts. Their demos never attempt persistence.
- `make_fusion_video_o3d.py --display stream` (new): replicates that look. In stream mode the
  panel pair is [streaming current-frame | naive accumulation] — the motivation shot for our
  demo: streaming is clean but memory-less; accumulation remembers but ghosts; the registry
  layer (Mode 3) = remember without ghosting.
  Output: `results/ex004_auto/fusion_m1_vs_m2_o3d_follow_stream.mp4`.
- Implication for Day-3 phone videos: shoot with GENTLE camera motion (already in the
  shooting instructions); the Mode-1-vs-Mode-3 money shot depends on it.
