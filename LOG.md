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
