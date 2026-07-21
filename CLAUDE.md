# CLAUDE.md - RayMap3R Dynamic-Scene Demo

## Project Overview

RayMap3R (CUT3R fork, training-free) 上做 **instance-level scene-state maintenance** demo:
追蹤同一物體隨時間的位置變化(observed / unobserved 兩種 regime),維護 persistent scene state。

**Roadmap 的單一真相來源是 `HANDOFF_raymap3r_demo.md` — 每個 session 開始先讀它。**
本檔只放跨 session 必須記住的硬規則與 code pointers。

## Repo / Remote

- `origin` = 私人 repo `george09270927/raymap3r_george`(開發推這裡)
- `upstream` = 上游 `Brack-Wang/raymap3r`(只 fetch,不 push)

## Machine

- **MacBook 只做編輯與閱讀,不跑 inference。** 跑 inference 用 **yoshi**(直連 `ssh yoshi`;RTX 4090 24GB,Ubuntu 22.04,CUDA toolkit `/usr/local/cuda-12.1`)。實測 VRAM ~8.3 GB @ 512 long-side。
- yoshi 上 repo:`~/Warehouse/raymap3r_george`(5.5T 資料碟)。
- 環境啟用:`source ~/miniconda3/etc/profile.d/conda.sh && conda activate raymap3r`(由 `scripts/setup_env_yoshi.sh` 建立;含 torch 2.1.1+cu121、transformers 4.45.2、accelerate、omegaconf、curope extension)。
- Workflow:Mac 編輯 → commit + push → yoshi `git pull` → ssh 執行;runs 輸出在 yoshi `results/`(gitignored)。
- 備援:zeldajr Slurm(quinn/rapunzel 3090);nano4 已開始收費,只有大規模 batch 才考慮(用前先讀官方 hackmd 與 OpenYOLO3D 的 `scripts/nchc/nano4.md`)。

## Boundaries — 絕對不做(HANDOFF §5)

- 不碰 `src/croco/` 或深層 DUSt3R internals(繼承的 engine code)
- 不 retrain / fine-tune 任何東西
- 暫不調 gate 超參數(`alpha_gate_lambda`、`alpha_ema_tau` 等)— 只記錄預設值;tau 實驗要等 go/no-go 之後
- 每次 run 的完整 command + output dir 記進 `LOG.md`(結果要給 mentor 看)

## Long-Running Rules

- 跑完整 sequence 之前,先用 5-10 frames 小測試:確認輸出檔案格式正確、下游讀得動,再開全量。
- 若用 nano4 / Slurm cluster:遵守 global CLAUDE.md 的 HPC 規則(login node 禁算、sbatch 必配 Telegram monitor、腳本進 git)。

## Key Code Pointers(verified 2026-07 clone)

| 位置 | 內容 |
|------|------|
| `src/dust3r/model.py` `_compute_alpha_state_from_static`(約 line 986-1070) | per-pixel dynamic map:`alpha_img = sigmoid(scale * (tau - delta))`,(B,1,H,W),目前算了但沒存 |
| `src/dust3r/model.py` 約 line 326-331 | gate 超參數預設值(alpha_gate_lambda=0.9, alpha_gate_wmin=0.15, alpha_ema_tau=5.0, coverage_adapt_k=0.5, small_step_c=0.5) |
| `src/dust3r/model.py` line 4 | dev comment:gate 對 depth 較好、**對 pose 較差** — registration 吃 pose,要驗證 gate on/off 的 pose 品質 |
| `infer.py` / `src/dust3r/inference.py` `inference_recurrent_lighter` | recurrent inference 主路徑;alpha_img 要從這裡 thread 出去存檔 |
| `infer.py` `--model_update_type` | **此 flag 在 recurrent 路徑上無效(2026-07-22 實測,兩值輸出 bit-identical)**:`forward_recurrent_lighter` 會重設它並交給內部 router 決定。gate A/B 要先做強制 regime 的小 patch |
| `src/dust3r/model.py` rotation router(約 1664-1674, 1902-1909, 1958-1970) | ~20 frames warm-up 量 camera rotation,median < 2°/frame → `xattn`(gated),否則留 `cut3r`;每序列只決定一次。`res["alpha_img"]` 在約 line 1888 已放進 per-frame pred |
| `infer.py` 輸出 | `depth/*.npy`, `conf/*.npy`, `color/*.png`, `camera/*.npz`, `poses_c2w.npy`, `trajectory.txt`, `pointcloud.ply`, `summary.json` |

## Weights

- Training-free,直接用 CUT3R checkpoint:`weights/cut3r_512_dpt_4_64.pth`(從 CUT3R repo releases 下載)
