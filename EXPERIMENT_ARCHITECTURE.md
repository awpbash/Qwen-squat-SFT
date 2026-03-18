# VLM Squat Coach: Experiment Architecture & Pipeline

> Comprehensive documentation of the QwenSFT experiment pipeline, from raw data to thesis figures.
> Updated: 2026-03-16 (v3.1 final results)

---

## 1. Overview

This project investigates whether fine-tuning a Vision-Language Model (VLM) with tool-augmented agentic reasoning improves squat form classification over zero-shot baselines. The pipeline trains Qwen2.5-VL-7B-Instruct with LoRA on the QEVD squat dataset, evaluates 7 experiment conditions across controlled and wild domains, and produces publication-ready statistical analysis.

### Pipeline Phases

```
Phase 0 (CPU)  --- Data Preparation ------------------------------------------
  clean_labels.py -> create_splits.py -> generate_agentic_training_data.py
                                      -> prepare_reddit_eval.py

Phase 1 (GPU)  --- Training ---------------------------------------------------
  finetune_video.sh (E2: Standard SFT)
  finetune_video_agentic.sh (E5b: Agentic SFT)

Phase 2 (GPU)  --- Evaluation: Controlled Domain (QEVD) -----------------------
  E1 -> E2 -> E3 -> E4 -> E5 -> E5b -> E5c  (413 test samples)

Phase 3 (GPU)  --- Evaluation: Wild Domain (Reddit) ---------------------------
  E1 -> E2 -> E3 -> E4 -> E5 -> E5b -> E5c  (70 videos)

Phase 4 (CPU)  --- Analysis & Figures -----------------------------------------
  eval_transfer.py -> eval_coaching.py -> eval_statistical.py -> thesis_tables.py
```

Orchestrated by: `run_all.sh` (supports `--phase N`, `--dry`, `--skip-training`)

---

## 2. Label Taxonomy

12 labels across 5 groups, defined in `constants.py`:

| Group | Labels | Cardinality | Notes |
|-------|--------|-------------|-------|
| **Stance** | shoulder-width, narrow, wide, plie | Exactly 1 | Mutually exclusive |
| **Depth** | shallow, 90 degrees, over 90 degrees | Exactly 1 | Mutually exclusive |
| **Form Issues** | back not straight, knees over toes, insufficient | 0-3 | Multi-label, often empty |
| **Variant** | hold | 0-1 | Pause at bottom position |
| **Visibility** | not visible | Exclusive | Nullifies all other fields |

Output format (JSON):
```json
{
  "stance": "shoulder-width",
  "depth": "over 90 degrees",
  "form_issues": ["knees over toes"],
  "variant": null,
  "visible": true
}
```

---

## 3. Data Pipeline

### 3.1 Raw to Clean (`clean_labels.py`)

| | Count |
|--|-------|
| Input (raw QEVD) | 4,308 |
| Fixed (label conflicts resolved) | 527 |
| Dropped (unsalvageable) | 276 |
| **Output** | **4,032** |

**Cleaning rules** (priority order):
1. `not visible` is exclusive - removes all other labels
2. `plie` overrides other stances - removes shoulder-width/narrow/wide
3. Multi-stance - keep non-default (narrow > wide > shoulder-width)
4. Multi-depth - keep most specific (over 90 > shallow > 90)
5. `starting late` - drop entirely (no visual grounding)

### 3.2 Splits + Oversampling (`create_splits.py`)

**Stratified split** (80/10/10) by rarest-label-per-sample, ensuring minority labels represented in all splits. No data leakage (verified: 0 video overlap across splits).

**Per-group balanced oversampling** (train only):
- For each sample: weight = max over its labels of `(group_max / label_count)`
- Whole-sample duplication preserves multi-label correlations
- `--max_ratio 20.0` caps duplication per sample
- **Form-issue balancing**: post-step duplicates form-issue samples to reach 50:50 vs no-form

| Split | Samples | Unique Videos | Notes |
|-------|---------|---------------|-------|
| Train | 11,180 | 3,220 | Aggressively oversampled + form-balanced |
| Val | 399 | 399 | Natural distribution |
| Test | 413 | 413 | Natural distribution |

**Imbalance reduction**:
| Group | Before (raw) | After (oversampled) |
|-------|-------------|----------------------|
| Stance max:min | 11.4x | 1.7x |
| Depth max:min | 2.1x | 1.9x |
| Form vs no-form | 40:60 | 50:50 |

### 3.3 Agentic Data Generation (`generate_agentic_training_data.py`)

Creates training data for the agentic pipeline (E5b) with per-field confidence labels.

**Confidence assignment** (measurement-grounded):
```
For each field (stance, depth, form_issues, variant, visible):
  Run pose_tool on video -> get suggested label
  If pose suggestion == ground truth -> confidence = LOW (tool helps)
  If pose suggestion != ground truth -> confidence = HIGH (tool misleads)
```

**Tool decision**: If ANY field is LOW -> `INVOKE_TOOL`, else `NO_TOOL_NEEDED`

**Outputs**:
- `new_qevd_mrq_agentic_train.json` - all agentic format
- `new_qevd_mrq_mixed_train.json` - 50/50 standard + agentic (size-matched to standard train)

### 3.4 Reddit Wild Set (`prepare_reddit_eval.py`)

70 in-the-wild squat videos from Reddit biomechanics communities.

**Label mapping** (Reddit to QEVD):
| Reddit Label | QEVD Equivalent |
|-------------|-----------------|
| insufficient_depth | shallow |
| back_rounding, forward_lean, butt_wink, torso_collapse | back not straight |
| knee_valgus | knees over toes |
| heel_lift, loss_of_bracing, unstable_feet | *(no QEVD equivalent)* |

---

## 4. Pose Measurement Tool (`pose_tool.py`)

MediaPipe Pose Landmarker providing objective body measurements for squat analysis.

### Measurements Computed

| Measurement | Method | Classification Thresholds |
|------------|--------|--------------------------|
| **Knee angle** | 3-point angle: hip-knee-ankle (avg L+R) | >110 = shallow, 85-110 = 90, <85 = over 90 |
| **Back deviation** | Angle between shoulder-hip line and vertical | >20 = back not straight |
| **Stance ratio** | ankle_distance / hip_distance | <0.8 = narrow, 0.8-1.5 = shoulder-width, 1.5-2.0 = wide, >2.0 = plie |
| **Knee-over-toe** | Horizontal offset of knee past ankle (normalized) | >0.08 = knees over toes |
| **Hold duration** | Consecutive frames within 5 deg of min knee angle | >1.0s = hold variant |
| **Visibility** | Average visibility of 8 key landmarks | <0.3 = not visible |

**Output**: JSON with raw measurements + `suggested_labels` dict.
**Caching**: Results stored in `pose_cache.json` to avoid recomputation.

---

## 5. Model Architecture

### Base Model
**Qwen2.5-VL-7B-Instruct** - 7B-parameter vision-language model

```
Qwen2.5VLForConditionalGeneration
|-- visual                          <-- Vision tower (FROZEN)
|   |-- blocks (ViT layers)
|   +-- merger                      <-- Vision-language connector (trainable, LR=5e-6)
+-- language_model                  <-- LLM backbone (frozen, LoRA adapters at LR=1e-5)
    |-- layers (transformer blocks)
    |   +-- [LoRA A/B matrices on all Linear layers]
    +-- lm_head (excluded from LoRA)
```

### LoRA Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Rank | 32 | Low-rank adaptation dimension |
| Alpha | 32 | Scaling factor (alpha/rank = 1.0) |
| Dropout | 0.05 | Applied to LoRA layers |
| Target modules | All Linear + Embedding | Except lm_head, embed_tokens, visual |
| Bias | none | No bias terms in LoRA |

### Two-Tier Learning Rates

| Component | Learning Rate | Role |
|-----------|--------------|------|
| LLM LoRA | 1e-5 | Main adaptation - learns squat label patterns |
| Merger | 5e-6 | Adjusts vision-to-language projection |
| Vision tower | N/A | **Frozen** - preserves pretrained visual features |

### Training Configuration (v3.1)

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Global batch size | 64 (4/device x 4 GPUs x grad_accum=4) |
| Loss function | Standard CE (Liger-fused) |
| Model selection | **Macro F1** on val set (generation-based eval) |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| Scheduler | Cosine with 5% warmup |
| Weight decay | 0.05 |
| Video FPS | 4 |
| Video max pixels | 128x32x32 = 131,072 |
| DeepSpeed | ZeRO Stage 2 |
| Precision | bf16 |
| Gradient checkpointing | Enabled |

### Training Curve (Val Set, 399 samples)

```
Epoch 1:  macro_f1 = 0.225   loss = 0.068
Epoch 2:  macro_f1 = 0.362   loss = 0.046
Epoch 3:  macro_f1 = 0.391   loss = 0.043
Epoch 4:  macro_f1 = 0.402   loss = 0.044
Epoch 5:  macro_f1 = 0.420   loss = 0.044  <-- best checkpoint
```

---

## 6. Experiment Matrix

### 6.1 Seven Experiments

| ID | Name | Training | Inference Mode | Tool Use | Key Question |
|----|------|----------|---------------|----------|-------------|
| **E1** | Zero-shot | None | Direct VLM | None | Baseline capability? |
| **E2** | SFT | Standard SFT | Direct VLM | None | Does fine-tuning help? |
| **E3** | Pose-only | None | Rule-based | MediaPipe rules | How far do measurements go? |
| **E4** | Always-tool | Standard SFT | VLM + Pose + VLM | Always invoked | Does tool always help? |
| **E5** | Agentic (untrained) | None | ReAct loop | Model decides | Can base model use tools? |
| **E5b** | Agentic (trained) | Agentic SFT | ReAct loop | Model decides | Does agentic training help? |
| **E5c** | Agentic (SFT-only) | Standard SFT | ReAct loop | Model decides | Is agentic data needed? |

### 6.2 Evaluation Domains

Each experiment evaluated on **both**:
- **Controlled** (QEVD): 413 test samples, gym-quality video, full label coverage
- **Wild** (Reddit): 70 videos, in-the-wild, limited label coverage (3 active labels)

### 6.3 Inference Architectures

#### Direct VLM (E1, E2)
```
Video -> VLM -> JSON classification
```

#### Pose-Only Rules (E3)
```
Video -> MediaPipe -> Rule thresholds -> JSON classification
```

#### Always-Tool Two-Pass (E4)
```
Video -> VLM (Pass 1: initial classification)
      -> MediaPipe (measurements)
      -> VLM (Pass 2: refine with measurements + generate coaching)
```

#### Agentic ReAct Loop (E5, E5b, E5c)
```
Video -> VLM (Pass 1: assess + confidence per field + tool decision)
      |-- If INVOKE_TOOL:
      |     -> MediaPipe (measurements)
      |     -> VLM (Pass 2: refine + coaching)
      +-- If NO_TOOL_NEEDED:
            -> VLM (Pass 2: coaching only)
```

---

## 7. Evaluation Metrics

### 7.1 Classification Metrics (`eval_unified.py`, `eval_pose_only.py`)

| Metric | Scope | Description |
|--------|-------|-------------|
| **Macro F1 (active)** | Primary | Mean F1 across labels with ground-truth support > 0 |
| Micro F1 | Secondary | Global TP/FP/FN aggregated across all labels |
| Per-label F1 | Diagnostic | Individual precision/recall/F1 for each of 12 labels |
| Stance accuracy | Group | Exact match on stance field |
| Depth accuracy | Group | Exact match on depth field |
| Geometric F1 | Group | Mean F1 over stance + depth labels |
| Holistic F1 | Group | Mean F1 over form issue labels |
| Label coverage | Diagnostic | Fraction of 12 labels predicted at least once |
| Bootstrap 95% CI | Uncertainty | 5,000 resamples for all metrics |

### 7.2 Transfer Analysis (`eval_transfer.py`)

- Per-label F1 gap: `F1_controlled - F1_wild`
- Scoped to labels with >=5 samples in both domains
- Bootstrap CI on the gap itself

### 7.3 Coaching Quality (`eval_coaching.py`)

| Metric | Description |
|--------|-------------|
| Issue recall | % of detected issues mentioned in coaching |
| Issue precision | % of mentioned issues actually present |
| Hallucination rate | % of coaching points for non-existent issues |
| Specificity score | 0=generic, 1=issue identified, 2=actionable cue |

### 7.4 Statistical Significance (`eval_statistical.py`)

- Pairwise McNemar's test (exact-match accuracy)
- Per-label McNemar's (binary correct/incorrect)
- Holm-Bonferroni correction for multiple comparisons
- Cohen's d effect sizes for transfer gaps

---

## 8. Results (v3.1)

### 8.1 Main Results

| Experiment | Controlled F1 | 95% CI | Wild F1 | Transfer Gap | Stance Acc | Depth Acc | Holistic F1 | Coverage |
|-----------|:------------:|:------:|:------:|:-----------:|:---------:|:--------:|:----------:|:-------:|
| E1 zero-shot | 0.130 | [0.115, 0.145] | 0.150 | -0.020 | 0.211 | 0.230 | 0.088 | 0.58 |
| **E2 SFT** | **0.327** | [0.293, 0.358] | 0.178 | 0.148 | **0.608** | **0.462** | 0.235 | 0.83 |
| E3 pose-only | 0.246 | [0.220, 0.271] | 0.167 | 0.079 | 0.269 | 0.373 | 0.073 | 0.83 |
| E4 always-tool | 0.270 | [0.240, 0.299] | 0.275 | **-0.005** | 0.559 | 0.235 | 0.208 | 0.92 |
| E5 agentic (untrained) | 0.075 | [0.057, 0.093] | 0.130 | -0.055 | 0.111 | 0.179 | 0.024 | 0.33 |
| **E5b agentic (trained)** | 0.246 | [0.219, 0.271] | **0.289** | -0.042 | 0.334 | 0.320 | **0.294** | 0.92 |
| E5c agentic (SFT-only) | 0.042 | [0.022, 0.063] | 0.062 | -0.020 | 0.053 | 0.179 | 0.046 | 0.50 |

### 8.2 Per-Label F1 (Controlled Domain)

| Label | E1 | E2 | E3 | E4 | E5 | E5b | E5c |
|-------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| **Stance** | | | | | | | |
| shoulder-width | 0.322 | **0.729** | 0.312 | 0.681 | 0.000 | 0.471 | 0.000 |
| narrow | 0.204 | 0.400 | 0.410 | **0.447** | 0.361 | 0.413 | 0.031 |
| wide | 0.139 | **0.375** | 0.012 | 0.337 | 0.236 | 0.165 | 0.091 |
| plie | 0.000 | **0.447** | 0.370 | 0.408 | 0.000 | 0.238 | 0.000 |
| **Depth** | | | | | | | |
| shallow | 0.358 | **0.487** | 0.425 | 0.416 | 0.237 | 0.395 | 0.148 |
| 90 degrees | 0.105 | **0.575** | 0.288 | 0.077 | 0.000 | 0.264 | 0.000 |
| over 90 degrees | 0.000 | 0.202 | **0.528** | 0.024 | 0.000 | 0.048 | 0.000 |
| **Form Issues** | | | | | | | |
| back not straight | 0.264 | **0.278** | 0.097 | 0.270 | 0.071 | 0.200 | 0.061 |
| knees over toes | 0.000 | 0.188 | 0.122 | 0.169 | 0.000 | **0.354** | 0.077 |
| insufficient | 0.000 | 0.239 | 0.000 | 0.185 | 0.000 | **0.329** | 0.000 |
| **Other** | | | | | | | |
| hold | 0.170 | 0.000 | **0.389** | 0.232 | 0.000 | 0.081 | 0.095 |
| not visible | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 8.3 Per-Label F1 (Wild Domain)

Only 3 labels have ground-truth support in the wild set: shallow, back not straight, knees over toes.

| Label | E1 | E2 | E3 | E4 | E5 | E5b | E5c |
|-------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| shallow | 0.000 | 0.091 | 0.000 | 0.393 | 0.080 | **0.449** | 0.100 |
| back not straight | **0.450** | 0.444 | 0.500 | 0.432 | 0.311 | 0.417 | 0.087 |
| knees over toes | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 8.4 Statistical Significance (McNemar's, Holm-Bonferroni corrected)

Key pairwise comparisons (controlled domain):

| Comparison | chi2 | p-value | Significant (corrected) |
|-----------|:----:|:-------:|:----------------------:|
| E1 vs **E2** | 30.63 | <0.001 | **Yes** |
| E1 vs E3 | 12.00 | <0.001 | **Yes** |
| **E2** vs E3 | 4.06 | 0.044 | No (after correction) |
| **E2** vs E4 | 20.05 | <0.001 | **Yes** |
| **E2** vs E5b | 16.00 | <0.001 | **Yes** |
| E3 vs E4 | 0.39 | 0.532 | No |
| E4 vs E5b | 1.24 | 0.265 | No |

### 8.5 Transfer Gap Analysis

| Experiment | Controlled F1 | Wild F1 (active) | Gap | 95% CI | Interpretation |
|-----------|:------------:|:----------------:|:---:|:------:|---------------|
| E1 zero-shot | 0.130 | 0.150 | -0.020 | [-0.065, 0.030] | No gap (CI includes 0) |
| **E2 SFT** | 0.327 | 0.178 | **0.148** | [0.065, 0.221] | **Significant gap** |
| E3 pose-only | 0.246 | 0.167 | 0.079 | [0.031, 0.132] | Moderate gap |
| **E4 always-tool** | 0.270 | 0.275 | **-0.005** | [-0.066, 0.064] | **No gap** |
| E5b agentic | 0.246 | 0.289 | -0.042 | [-0.114, 0.038] | No gap (CI includes 0) |

---

## 9. Key Findings

### Finding 1: SFT dramatically improves controlled-domain performance
E2 (0.327) is **2.5x the zero-shot baseline** (0.130) and the best controlled-domain model. It achieves the highest stance accuracy (0.608) and depth accuracy (0.462), with strong per-label F1 across 10 of 12 labels. The improvement over E1 is statistically significant (p<0.001).

### Finding 2: Tool-augmented models generalize best to wild domain
E5b agentic trained achieves the **best wild-domain F1** (0.289), outperforming E2 (0.178) despite lower controlled performance. The pose tool provides domain-invariant measurements that transfer across video quality.

### Finding 3: The pose tool eliminates the transfer gap
E4 always-tool has a transfer gap of **-0.005** (CI includes 0). It performs identically on controlled and wild data. E2 has the largest gap (0.148, CI excludes 0). Models using the tool (E4, E5b) generalize; models without it (E2) overfit to controlled-domain appearance.

### Finding 4: E5b excels at form issue detection
E5b achieves holistic_avg_f1 = **0.294** (best), with standout knees_over_toes (0.354) and insufficient (0.329). The agentic confidence mechanism helps the model attend to subtle form issues that other models miss.

### Finding 5: Agentic-specific training data is essential
E5c (standard SFT in agentic pipeline) scores 0.042, confirming that agentic reasoning requires purpose-built training data. E5 (untrained base) similarly fails (0.075).

### Finding 6: Rule-based measurement excels on measurement-friendly labels
E3 achieves the best over_90_degrees (0.528) and hold (0.389) F1. These are labels where objective joint-angle and temporal measurements are most reliable. E2 scores 0.000 on hold.

### Finding 7: Complementary strengths across methods
No single method wins everywhere. E2 dominates stance/depth, E5b leads form issues, E3 excels on measurement-amenable labels, and E4 achieves the best domain stability. This motivates ensemble or cascaded approaches.

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Frozen vision tower** | Unfreezing on ~4K videos caused catastrophic forgetting (v2 lesson) |
| **LoRA scaling = 1.0** | Scaling=2.0 in v2 caused over-aggressive weight updates |
| **LR = 1e-5** | 5e-5 in v2 caused overfitting by epoch 2; 1e-5 allows steady improvement |
| **Standard CE + Liger** | Token-level focal loss doesn't address label imbalance; Liger gives ~30% speedup |
| **Non-LoRA state dict loading** | Trained merger weights must load alongside LoRA adapter at eval |
| **Left-pad generation** | Decoder-only models require left-padding for correct generation |
| **Form-issue balancing (50:50)** | 60% no-form samples caused model to always predict `[]` |
| **FPS=4** | FPS=2 missed bottom squat position; FPS=4 doubles temporal resolution |
| **Macro F1 model selection** | eval_loss is a poor proxy for classification quality |
| **ZeRO-2** | ZeRO-3 has known issues with LoRA parameter partitioning |
| **Bootstrap CI (N=5000)** | Small test set (413 samples) needs robust uncertainty estimation |

---

## 11. Version History

| Version | Key Changes | E2 Ctrl F1 | Best Wild F1 | Status |
|---------|-------------|:----------:|:----------:|--------|
| **v1** | Initial, 5 epochs, vanilla CE, eval_loss selection | 0.198 | - | Archived |
| **v2** | Oversampling, FPS=2, unfrozen vision tower | 0.171 | 0.218 (E5b) | Archived |
| **v3.1** | Frozen vision, LR=1e-5, alpha=32, Liger CE, left-pad fix, merger loading fix | **0.327** | **0.289** (E5b) | **Current** |

### v2 to v3.1: Root Causes of Improvement

1. **Vision tower frozen** - Pretrained ViT features preserved instead of destroyed by 4K-sample fine-tuning
2. **LR reduced 5x** (5e-5 -> 1e-5) with scaling halved (2.0 -> 1.0) - Prevents catastrophic forgetting
3. **Merger weights loaded at eval** - `non_lora_state_dict.bin` now loaded alongside LoRA adapter
4. **Left-padding for generation** - Correct padding for decoder-only model evaluation
5. **Macro F1 model selection** - Best checkpoint chosen by classification quality, not token-level loss

---

## 12. Output Structure

```
QwenSFT/
|-- data/QEVD-FIT-300k-Part-{1-4}/    # Raw video data (298K videos)
|-- squat_videos/                       # Symlinked/processed videos
|-- archive/
|   |-- v1_initial_experiments/         # v1 results (archived)
|   +-- reddit_eval/export_pack/        # Reddit source videos
|
|-- new_qevd_mrq_clean.json            # 4,032 cleaned samples
|-- new_qevd_mrq_train.json            # 11,180 oversampled train
|-- new_qevd_mrq_val.json              # 399 val (natural)
|-- new_qevd_mrq_test.json             # 413 test (natural)
|-- new_qevd_mrq_mixed_train.json      # 11,180 mixed (50/50 standard+agentic)
|-- reddit_eval_test.json               # 70 wild test
|
|-- Qwen-VL-Series-Finetune/
|   |-- scripts/
|   |   |-- finetune_video.sh           # E2 training config
|   |   |-- finetune_video_agentic.sh   # E5b training config
|   |   +-- zero2.json                  # DeepSpeed ZeRO-2 config
|   |-- output/
|   |   |-- sft_lora_v3/               # E2 adapter + checkpoints
|   |   +-- sft_lora_agentic_v3/       # E5b adapter + checkpoints
|   +-- src/
|       |-- trainer/sft_trainer.py      # Custom trainer (gen-based eval, 2-tier LR)
|       |-- train/train_sft.py          # Entry point (macro F1 compute_metrics)
|       |-- dataset/sft_dataset.py      # Video/image data loading
|       +-- params.py                   # All training arguments
|
|-- eval_results/
|   |-- E1_zero_shot/{controlled,wild}/ # metrics.json, predictions.jsonl
|   |-- E2_sft/...
|   |-- E3_pose_only/...
|   |-- E4_always_tool/...
|   |-- E5_agentic_untrained/...
|   |-- E5b_agentic_trained/...
|   |-- E5c_agentic_sft/...
|   |-- transfer_analysis/              # Transfer gap results
|   +-- statistical_summary/            # McNemar's + CIs
|
|-- thesis_figures/                     # Publication-ready outputs
|   |-- fig1-7 (.pdf + .png)           # 300 DPI figures
|   +-- table2-7 (.tex + .md)          # LaTeX + Markdown tables
|
+-- run_all.sh                          # Master orchestration script
```
