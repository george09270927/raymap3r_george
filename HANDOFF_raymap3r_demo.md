# HANDOFF: RayMap3R Dynamic-Scene Demo (Week 1)

> Context: George (NTHU ELSA Lab; Woven by Toyota CV intern from Aug 2026) is building a quick demo for his internship research direction: **instance-level scene-state maintenance in dynamic scenes** — track the same object over time, detect that "the chair moved from A to B" (both while watching AND while not watching), and update a persistent scene state / digital twin.
> This doc was prepared by Claude (claude.ai project) after full literature triage and a first read of the RayMap3R codebase. Trust the file/line pointers below; they were verified against a fresh clone (2026-07).

---

## 0. First task: pick the machine

The MacBook is for editing only — inference needs CUDA (repo pins `torch==2.1.1 cu121`) and ~9–10 GB VRAM (paper reports 9.2–9.4 GB constant memory).

1. Read `~/.ssh/config`. Candidate hosts include lab boxes (e.g. `zeldajr`, `yoshi`) and the NCHC `nano4` H200 cluster.
2. For each reachable host: `nvidia-smi` (VRAM ≥ 12 GB comfortable), check disk, CUDA version, whether you can hold an interactive session.
3. **Prefer an interactive lab box for week 1.** nano4 is a batch cluster with 48-hour job limits — right for later training runs, awkward for tight edit-run-look loops. Recommend a choice to George before proceeding.
4. Workflow: clone + edit + run on the chosen remote host (Claude Code over SSH or on-host). Keep the MacBook copy only if George wants local reading.

## 1. Setup (Day 1)

```bash
git clone https://github.com/Brack-Wang/raymap3r.git
cd raymap3r
# python env per README; torch 2.1.1 cu121
pip install -r requirements.txt
# Weights: RayMap3R is training-free and reuses the CUT3R checkpoint.
# Download cut3r_512_dpt_4_64.pth from the CUT3R repo releases → place at:
#   weights/cut3r_512_dpt_4_64.pth
```

Then run the official demo end-to-end on their sample input:
```bash
# CORRECTED 2026-07-22 (verified against infer.py argparse; the original --input flag does not exist):
python infer.py --video <clip.mp4> --output_dir results/smoke_test --model_update_type xattn
# --frames_dir <dir> for an image folder; --num_frames caps frames (default 30, 0 = all)
```
**IMPORTANT: `--model_update_type` defaults to `cut3r` = NO-gate baseline.** `xattn` enables the
attention + RayMap3R alpha gate — the README quickstart omits the flag, so copying it runs the
baseline. This same flag is the gate on/off switch for the §3 pose-quality comparison.

`infer.py` is headless; outputs go to `--output_dir`: `depth/*.npy`, `conf/*.npy`, `color/*.png`, `camera/*.npz` (pose + intrinsics), `poses_c2w.npy`, `trajectory.txt` (TUM-style xyz), fused `pointcloud.ply`, `summary.json`.

✅ Day-1 exit criterion: fused ply + per-frame depth/pose exist for the sample video. (This also validates the CUT3R inference path — the repo IS a CUT3R fork; `src/dust3r/model.py` = CUT3R "model_ori" + a static RayMap branch.)

## 2. Patch: expose the per-pixel dynamic map (Day 2)

The dynamic map is computed internally but **not saved**. Recon results:

- Computation lives in `src/dust3r/model.py :: _compute_alpha_state_from_static` (≈ lines 986–1070):
  - `delta` = relative depth discrepancy between main branch and RayMap-only branch
  - `alpha_img = sigmoid(scale * (tau - delta))` → **(B, 1, H, W) per-pixel static weight in [0,1]** (confidence-weighted, 3×3 avg-pool smoothed); code comments call it an "optional return"
  - It is pooled onto state tokens as `alpha_state` to gate memory: `s_t = s_{t-1} + alpha ⊙ Δs_t`
- Task: thread `alpha_img` through the recurrent inference outputs (`inference_recurrent_lighter` in `infer.py` / `src/dust3r/inference.py`) and add a save in `save_results` → `alpha/{i:06d}.npy`. Expect ~10–20 lines of plumbing.
- Also dump a per-frame heatmap overlay video (matplotlib/cv2) for eyeballing.

Gate hyperparameters (all exposed at `model.py` ≈ lines 326–331 — do NOT tune yet, but note them):
`alpha_gate_lambda=0.9`, `alpha_gate_wmin=0.15`, **`alpha_ema_tau=5.0`** (EMA window — this is the knob controlling how fast a moved-then-stopped object gets re-accepted into memory; it is the first research dial later), `coverage_adapt_k=0.5`, `small_step_c=0.5`.

✅ Day-2 exit criterion: heatmap video of `1 - alpha_img` on the sample sequence looks like a plausible motion map.

## 3. Go/no-go experiment: the two regimes (Day 3) ← MOST IMPORTANT

The paper only evaluates continuously-moving subjects (DAVIS/Sintel/TUM/Bonn). Our use case is **briefly-moved furniture**. Mechanism analysis predicts RayMap3R's signal is position-based ("differs from remembered state"), not velocity-based, so it should fire on both regimes — but this is an untested hypothesis. Test it before building anything on top.

Capture two phone videos indoors (George provides, or use any RGB clip):
- **(a) Observed motion**: camera watches while a chair/box is pushed ~1 m, then stops.
- **(b) Unobserved change**: camera pans away, an object is moved off-screen, camera pans back.

Check in the alpha heatmaps:
- (a): dynamic response during motion; decays after the object stops (EMA, tau=5.0).
- (b): on return, transient response at BOTH the old location (memory says object, image says empty) and the new location (memory says empty, image says object), then decay as memory accepts the new state.

✅ Go: both regimes produce a usable transient signal → proceed to Day 4.
❌ No-go: regime (b) silent or hopelessly noisy → escalate to George; fallback paths: (i) explicit registry-diff for unobserved regime (compare SAM2 instances vs registry on revisit), (ii) Pi3MOS-SLAM masks for observed regime only.

Also log during Day 1–3: pose quality with gate on vs off. `model.py` line 4 has an honest dev comment: "works better than baseline model_ori on depth, **worse on pose**" — contradicts the paper's framing; our registration consumes poses, so verify. (Gate-off = `--model_update_type cut3r` (default) ≈ vanilla CUT3R; gate-on = `--model_update_type xattn`. Verified 2026-07-22 in `infer.py` — same run command, one flag.)

## 4. Instances + registry (Day 4–5)

1. Install SAM2 (`facebookresearch/sam2`). Usage pattern: **discovery is slow, tracking is fast (~44 FPS)** → run automatic mask generation every N frames (or on high change-coverage), propagate masks in between. Cost grows linearly with #objects — fine for demo scenes.
2. Per-instance motion flag: aggregate `(1 - alpha_img)` inside each SAM2 instance mask per frame → "instance k is moving / was changed".
3. Minimal registry (a dict is fine):
   ```
   registry[id] = { points_3d, pose (4x4 or (x,y,z,yaw)), state: static|moving|relocated, last_seen }
   ```
   Points come from lifting the SAM2 mask through the saved depth + camera per frame.
4. Rigid registration: when an instance's motion flag rises then settles, run Procrustes/ICP (open3d) between its registry points and its current points → Δpose; update registry.

✅ Week-1 deliverable: terminal line like `chair_03 moved: Δ=(0.9m, 0.2m, 15°)` + a before/after render (two plys or two colored snapshots of the fused cloud).

## 5. Boundaries — do NOT

- Do NOT build Pi3MOS-SLAM this week (DPVO + BA + Eigen build; it's the week-2+ parallel track ONLY IF masks prove insufficient).
- Do NOT touch `src/croco/` or deep DUSt3R internals — inherited engine code.
- Do NOT retrain or fine-tune anything.
- Do NOT tune gate hyperparameters yet (record defaults; tau experiments come after the go/no-go).
- Keep every run's command + output dir logged in a `LOG.md`; George will bring results to his mentor (HK).

## 6. Reference intel (for questions that may come up)

- RayMap3R: ECCV 2026, training-free, MIT license, 13.8 FPS vs CUT3R 19.7 on A100 @512.
- The demo's conceptual split: CUT3R = "where things are", RayMap3R alpha = "what changed/moves", SAM2 = "which object is which"; the registry + registration layer is OUR code and the only novel part of the demo.
- Known failure modes to watch: SAM2 identity switches after occlusion; metric-scale drift on long sequences (keep clips < 1–2 min); large dynamic coverage degrading pose (gate has coverage adaptation but verify).
- If a question needs paper-level detail (Khronos, O-SCD, Pi3MOS-SLAM, competitive landscape), defer to George — full readouts live in his Notion / claude.ai project.
