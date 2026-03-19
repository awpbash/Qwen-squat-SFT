# VLM Squat Coach: Fine-Tuning and Transfer Study

Fine-tuning **Qwen2.5-VL-7B-Instruct** with LoRA for multi-label squat form classification, with tool-augmented agentic inference and controlled-to-wild transfer analysis.

> Full paper: [RESULTS_REPORT.md](RESULTS_REPORT.md) | Architecture docs: [EXPERIMENT_ARCHITECTURE.md](EXPERIMENT_ARCHITECTURE.md)


## Results (v3.1)

![Macro F1 across all 7 experiments (controlled domain) with 95% bootstrap CIs](thesis_figures/fig1_macro_f1_controlled.png)

F1 values below are **macro F1 (active)**, averaged only over labels with ground-truth support > 0 in each domain (12 labels for controlled, 3 for wild). See the [metric definitions in the report](RESULTS_REPORT.md#0-metric-definitions-read-first) for details.

| Experiment | Controlled F1 | Wild F1 | Key Finding |
|-----------|:------------:|:------:|-------------|
| E1 Zero-shot | 0.130 | 0.150 | Baseline |
| **E2 SFT** | **0.327** | 0.178 | Best controlled (+151% over E1, p<0.001) |
| E3 Pose-only | 0.246 | 0.167 | Best on measurement-amenable labels |
| E4 Always-tool | 0.270 | 0.275 | Near-zero transfer gap (-0.005) |
| E5b Agentic trained | 0.246 | **0.289** | Best wild; best form issue detection |

Per-label breakdown: [Table 3](thesis_figures/table3_per_label.md) | Transfer gaps: [Table 4](thesis_figures/table4_transfer_gap.md) | Statistical tests: [Table 7](thesis_figures/table7_mcnemar.md)

**Key caveat:** Coaching text generation suffers 80-90% hallucination rates. The model produces specific, actionable, but often incorrect advice. See [coaching analysis](thesis_figures/table6_coaching.md) and Section 5.7 of the [report](RESULTS_REPORT.md#57-coaching-safety-implications).


## Research Questions

| # | Question | Answer |
|---|---------|--------|
| RQ1 | Does SFT improve over zero-shot? | **Yes**, 2.5x improvement (p<0.001) |
| RQ2 | Does tool augmentation help? | **Nuanced**: hurts controlled, stabilises transfer |
| RQ3 | Does agentic training improve form detection? | **Suggestive**: best form issue F1, but confounded with data |
| RQ4 | What drives the transfer gap? | **Exploratory**: tool-augmented models show smaller gaps (N=70) |


## Prerequisites

**Hardware:**
- 4x GPUs with at least 24GB VRAM each (tested on A100 40GB)
- ~50GB disk for training checkpoints

**Data (not included in repo):**
- [QEVD dataset](https://github.com/Garfield-hr/QEVD) videos in `data/QEVD-FIT-300k-Part-{1-4}/`
- Reddit evaluation videos in `squat_videos/` (scraped manually, not redistributable)

**Environment:**
```bash
pip install transformers peft accelerate deepspeed
pip install mediapipe opencv-python numpy scipy
pip install matplotlib  # for thesis figures
```


## Quick Start

```bash
# Full pipeline (all phases)
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_all.sh

# Start from a specific phase
bash run_all.sh --phase 2

# Skip training (use existing checkpoints)
bash run_all.sh --skip-training

# Dry run (print commands only)
bash run_all.sh --dry
```

### Regenerating Training Data

The large training JSONs are not committed (see .gitignore). To regenerate from the committed clean/val/test splits:

```bash
# Step 1: Create oversampled train split from clean data
#   Input:  new_qevd_mrq_clean.json (requires raw QEVD data + clean_labels.py to produce)
#   Output: new_qevd_mrq_train.json (11,180 samples)
python create_splits.py

# Step 2: Generate agentic training data (requires videos on disk for pose analysis)
#   Input:  new_qevd_mrq_train.json
#   Output: new_qevd_mrq_agentic_train.json, new_qevd_mrq_mixed_train.json
python generate_agentic_training_data.py
```

Note: `new_qevd_mrq_clean.json` is also gitignored (8MB). To reproduce it from scratch, run `clean_labels.py` on the raw QEVD annotation files.


## Pipeline

![Experiment inference pipelines](thesis_figures/fig9_experiment_pipelines.png)

```
Phase 0 (CPU)   Data Preparation
  clean_labels.py -> create_splits.py -> generate_agentic_training_data.py
                                      -> prepare_reddit_eval.py

Phase 1 (GPU)   Training
  finetune_video.sh          (E2: Standard SFT)
  finetune_video_agentic.sh  (E5b: Agentic SFT)

Phase 2-3 (GPU) Evaluation (controlled + wild)
  eval_unified.py, eval_pose_only.py, inference_with_pose.py, inference_agentic.py

Phase 4 (CPU)   Analysis and Figures
  eval_transfer.py -> eval_coaching.py -> eval_statistical.py -> thesis_tables.py
```

Orchestrated by `run_all.sh` (supports `--phase N`, `--dry`, `--skip-training`).


## Model Architecture

![Model architecture with frozen/trainable components](thesis_figures/fig8_model_architecture.png)

| Component | Params | Trainable | LR |
|-----------|--------|-----------|-----|
| Vision tower (ViT) | ~400M | Frozen | - |
| Merger | ~10M | Full | 5e-6 |
| LLM | ~7B | LoRA (~25M, rank=32, alpha=32) | 1e-5 |

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 64 global (4/device x 4 GPUs x 4 grad_accum) |
| Loss | Standard CE (Liger-fused) |
| Model selection | Val macro F1 (generation-based) |
| FPS | 4 |
| DeepSpeed | ZeRO-2, bf16 |


## Label Taxonomy

12 labels across 5 groups:

| Group | Labels | Cardinality |
|-------|--------|-------------|
| Stance | shoulder-width, narrow, wide, plie | Exactly 1 |
| Depth | shallow, 90 degrees, over 90 degrees | Exactly 1 |
| Form Issues | back not straight, knees over toes, insufficient | 0-3 |
| Variant | hold | 0-1 |
| Visibility | not visible | Exclusive (nullifies all other fields) |


## Data

| Split | Samples | Notes |
|-------|---------|-------|
| Train | 11,180 | Oversampled from 3,220 unique videos |
| Val | 399 | Natural distribution |
| Test | 413 | Natural distribution |
| Wild (Reddit) | 70 | 3 active labels only (exploratory) |

Raw: 4,308 -> Cleaned: 4,032 (276 dropped, 527 conflict-resolved).
Oversampling reduces stance imbalance from 11.4x to 1.9x. See [Figure 2](thesis_figures/fig2_label_distribution.png) and [Figure 2b](thesis_figures/fig2b_oversampling_effect.png) for the per-label breakdown.


## Experiment Matrix

| ID | Name | Training | Inference | Tool |
|----|------|----------|-----------|------|
| E1 | Zero-shot | None | Direct VLM | None |
| E2 | SFT | Standard SFT | Direct VLM | None |
| E3 | Pose-only | None | Rule-based | MediaPipe |
| E4 | Always-tool | E2 model | VLM + Pose + VLM | Always |
| E5 | Agentic untrained | None | ReAct loop | Model decides |
| E5b | Agentic trained | Mixed SFT (50% standard + 50% agentic) | ReAct loop | Model decides |
| E5c | Agentic SFT-only | E2 model | ReAct loop | Model decides |

**Clean comparisons:** E1 vs E2 (SFT effect), E4 pass-1 vs pass-2 (tool effect), E5 vs E5b vs E5c (agentic training effect).


## Figures and Tables

| Output | Description |
|--------|-------------|
| [Fig 1: Macro F1 (controlled)](thesis_figures/fig1_macro_f1_controlled.png) | Main result, bar chart with 95% CIs |
| [Fig 2: Label distribution](thesis_figures/fig2_label_distribution.png) | Pre/post oversampling label counts |
| [Fig 2b: Oversampling effect](thesis_figures/fig2b_oversampling_effect.png) | Per-label distribution (%) with boost factors and imbalance ratios |
| [Fig 3: Per-label F1](thesis_figures/fig3_per_label_bar.png) | Controlled vs wild, per label |
| [Fig 4: Transfer gap](thesis_figures/fig4_transfer_gap.png) | Per-label gap by experiment |
| [Fig 5: Calibration](thesis_figures/fig5_calibration.png) | Tool invocation calibration (agentic experiments) |
| [Fig 6: F1 heatmap](thesis_figures/fig6_f1_heatmap.png) | Per-label x per-experiment heatmap |
| [Fig 7: Radar chart](thesis_figures/fig7_radar.png) | Multi-metric comparison |
| [Fig 8: Architecture](thesis_figures/fig8_model_architecture.png) | Model architecture diagram |
| [Fig 9: Pipelines](thesis_figures/fig9_experiment_pipelines.png) | Inference pipeline per experiment |
| [Table 2: Main results](thesis_figures/table2_main_results.md) | All metrics with CIs |
| [Table 3: Per-label F1](thesis_figures/table3_per_label.md) | Full per-label breakdown |
| [Table 4: Transfer gaps](thesis_figures/table4_transfer_gap.md) | Per-label transfer analysis |
| [Table 5: Tool calibration](thesis_figures/table5_calibration.md) | Per-field invoke rates and accuracy |
| [Table 6: Coaching quality](thesis_figures/table6_coaching.md) | Hallucination rates, recall, specificity |
| [Table 7: McNemar's tests](thesis_figures/table7_mcnemar.md) | Pairwise significance (Holm-Bonferroni corrected) |

LaTeX versions of all tables are available in `thesis_figures/*.tex`.


## File Structure

```
QwenSFT/
|-- run_all.sh                          # Master orchestration
|-- README.md                           # This file
|-- EXPERIMENT_ARCHITECTURE.md          # Architecture documentation
|-- RESULTS_REPORT.md                   # Full paper with analysis
|-- .gitignore
|
|-- clean_labels.py                     # Phase 0: Label cleaning
|-- create_splits.py                    # Phase 0: Splits + oversampling
|-- prepare_reddit_eval.py              # Phase 0: Reddit data prep
|-- generate_agentic_training_data.py   # Phase 0: Agentic data generation
|-- constants.py                        # Shared constants (label groups)
|-- pose_tool.py                        # MediaPipe pose analysis
|-- eda_plots.py                        # Exploratory data analysis plots
|
|-- eval_unified.py                     # Eval: standard VLM (E1, E2)
|-- eval_pose_only.py                   # Eval: pose rules (E3)
|-- inference_with_pose.py              # Eval: two-pass VLM+Pose (E4)
|-- inference_agentic.py                # Eval: agentic ReAct (E5, E5b, E5c)
|-- eval_transfer.py                    # Analysis: transfer gaps
|-- eval_coaching.py                    # Analysis: coaching quality
|-- eval_statistical.py                 # Analysis: McNemar's tests
|-- thesis_tables.py                    # Analysis: figures + LaTeX tables
|
|-- new_qevd_mrq_val.json              # Val (399)
|-- new_qevd_mrq_test.json             # Test (413)
|-- reddit_eval_test.json               # Wild eval (70 videos)
|
|-- eval_results/                       # Per-experiment metrics + predictions
|-- thesis_figures/                     # Publication-ready outputs (PNG/PDF/MD/TeX)
|
|   --- not committed (see .gitignore) ---
|
|-- new_qevd_mrq_clean.json            # Cleaned data (4,032) - regenerate with clean_labels.py
|-- new_qevd_mrq_train.json            # Train (11,180) - regenerate with create_splits.py
|-- new_qevd_mrq_agentic_train.json    # Agentic format - regenerate with generate_agentic_training_data.py
|-- new_qevd_mrq_mixed_train.json      # 50/50 mixed - regenerate with generate_agentic_training_data.py
|-- pose_cache.json                     # Cached pose measurements
|-- data/                               # Raw QEVD dataset (298K videos, ~179GB)
|-- squat_videos/                       # Reddit eval videos (~1.2GB)
|-- archive/                            # Old experiments v1, v2 (~1.2TB)
|
+-- Qwen-VL-Series-Finetune/           # Training framework (upstream: github.com/2U1)
    |-- src/                            # Framework source code
    |-- scripts/finetune_video.sh       # E2 training config
    |-- scripts/finetune_video_agentic.sh  # E5b training config
    +-- output/                         # Trained adapters (~235GB, not committed)
```
