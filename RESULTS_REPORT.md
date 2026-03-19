# Automated Squat Form Assessment Using Vision-Language Models with Tool-Augmented Inference

---

## Abstract

We investigate whether vision-language models (VLMs) can perform fine-grained squat form classification from video, and whether tool augmentation (invoking a pose estimation tool at inference time) improves accuracy or domain transfer. We fine-tune Qwen2.5-VL-7B-Instruct with LoRA and evaluate seven experimental conditions across a controlled gym-quality dataset (QEVD, N=413) and an exploratory in-the-wild Reddit dataset (N=70). Supervised fine-tuning (SFT) yields a 2.5x improvement over zero-shot (macro F1: 0.130 -> 0.327, p<0.001), but the best model still falls short of deployment-ready accuracy. Tool augmentation via MediaPipe pose estimation hurts controlled-domain accuracy (0.327 -> 0.270, p<0.001), degrading 34.9% of predictions while helping only 12.8%, yet produces a near-zero transfer gap between domains (-0.005, CI includes zero), compared to SFT's significant gap (+0.148, CI excludes zero). Agentic training (multi-step reasoning with tool invocation) improves form-issue detection (holistic F1: 0.294 vs 0.235) but is confounded with training data composition. Coaching text generation suffers 80-90% hallucination rates, posing safety concerns for deployment. No single method dominates across all label categories, motivating future ensemble approaches. We frame this work as a methodological study of VLM adaptation strategies for fine-grained video classification, and contribute an engineering analysis of six failure modes encountered during development.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Conclusion](#6-conclusion)
A. [Appendix: Figures and Tables Reference](#appendix-a-figures-and-tables-reference)

---

## 1. Introduction

Automated exercise form assessment has practical value for injury prevention and training efficiency, yet remains an open problem. Traditional approaches rely on pose estimation pipelines with hand-crafted rules [TODO: cite], which are interpretable but brittle across camera angles and body types. Recent vision-language models (VLMs) offer a potential alternative: end-to-end video understanding combined with structured output generation. However, whether VLMs can reliably assess fine-grained biomechanical properties from video, and whether tool augmentation can compensate for their visual limitations, has not been systematically studied.

This paper investigates these questions using squat form classification as a testbed. We fine-tune Qwen2.5-VL-7B-Instruct with LoRA and evaluate seven experimental conditions across two domains: a controlled gym-quality dataset (QEVD, N=413) and an exploratory in-the-wild Reddit dataset (N=70). Our experimental design isolates the contributions of supervised fine-tuning, pose-tool augmentation, and agentic (multi-step reasoning with tool use) training through controlled comparisons.

**On absolute performance.** The best macro F1 of 0.327 means the system is not yet reliable enough for unsupervised deployment. We frame this work as a methodological study of VLM adaptation strategies for fine-grained video classification, not as a production-ready coaching system. The results establish which directions improve performance and where the remaining gaps lie, informing future work toward deployable accuracy.

### 1.1 Research Questions

| ID | Question | Comparison | Status |
|----|---------|-----------|--------|
| **RQ1** | Does SFT improve squat classification over zero-shot? | E1 vs E2 (clean) | **Confirmed** (p<0.001) |
| **RQ2** | Does tool augmentation improve accuracy or transfer? | E2 vs E4, with pass-1 analysis | **Nuanced** (hurts controlled, helps transfer) |
| **RQ3** | Does agentic training improve form detection? | E5b vs E2 (confounded), E5b vs E5c (clean) | **Suggestive** (confounded with data) |
| **RQ4** | What drives the controlled-wild transfer gap? | Transfer analysis, all conditions | **Exploratory** (underpowered wild set) |

### 1.2 Contributions

1. **Systematic comparison of VLM adaptation strategies** for fine-grained video classification: zero-shot, SFT, tool-augmented, and agentic inference, with clean pairwise controls isolating each factor.
2. **Causal tool-effect analysis**: a pass-1 vs pass-2 decomposition within the tool-augmented pipeline that cleanly isolates the pose tool's contribution, independent of model or training differences.
3. **Transfer gap characterisation**: evidence that tool augmentation provides domain-invariant grounding, reducing the controlled-wild performance gap, with appropriate statistical caveats for the underpowered wild evaluation.
4. **Engineering contribution**: documentation of six failure modes in VLM fine-tuning (vision tower unfreezing, LoRA scaling, mode collapse, weight loading, padding direction, focal loss) that caused SFT to underperform zero-shot in early iterations.

> **Figure 1**: Macro F1 across all 7 experiments (controlled domain) with 95% bootstrap CIs.
>
> ![Figure 1: Macro F1 Controlled](thesis_figures/fig1_macro_f1_controlled.png)

---

## 2. Related Work

### 2.1 Exercise Form Assessment

Prior work on automated exercise assessment falls into two categories. **Pose-estimation approaches** extract skeletal keypoints (via OpenPose, MediaPipe, or learned detectors) and apply rule-based or classical ML classifiers to joint angles and trajectories [TODO: cite]. These systems achieve reasonable accuracy on constrained setups but are sensitive to camera angle, occlusion, and keypoint noise. We observe these limitations directly in our E3 (pose-only) baseline. **Deep learning approaches** use CNNs or transformer-based architectures trained end-to-end on exercise video [TODO: cite], typically with classification heads. These avoid explicit pose estimation but require large labelled datasets and do not generalise to coaching text generation.

Our work differs in using a **generative VLM** for structured multi-label classification. This preserves the model's ability to generate coaching text and reason about its predictions, at the cost of solving a harder problem (free-form generation with JSON parsing) than standard classification.

### 2.2 Vision-Language Models for Video Understanding

VLMs such as GPT-4V [TODO: cite], LLaVA-Video [TODO: cite], and Qwen2.5-VL [TODO: cite] have demonstrated strong zero-shot capabilities on video understanding benchmarks. However, their application to fine-grained biomechanical assessment, where the relevant visual signals (knee angle, spinal alignment) are subtle and domain-specific, has not been systematically evaluated. Our E1 (zero-shot) baseline establishes a reference point: Qwen2.5-VL-7B achieves macro F1 = 0.130 on squat classification without fine-tuning, indicating that general-purpose video understanding does not transfer directly to this domain.

### 2.3 Tool-Augmented Language Models

Recent work on tool-augmented LLMs [TODO: cite Toolformer, ReAct, etc.] has shown that language models can learn to invoke external tools (calculators, search engines, APIs) to compensate for their limitations. We extend this paradigm to vision: the VLM invokes a pose estimation tool (MediaPipe) to obtain joint angle measurements that augment its visual assessment. To our knowledge, this is the first systematic comparison of tool-augmented versus direct VLM inference for fine-grained video classification, with clean causal analysis (E4 pass-1 vs pass-2) isolating the tool's contribution.

---

## 3. Methodology

### 3.1 Task Formulation

We formulate squat form classification as a **structured multi-label prediction** task. Given a short video clip, the model outputs a JSON object:

```json
{
  "stance": "shoulder-width",          // 1 of 4 (mutually exclusive)
  "depth": "90 degrees",               // 1 of 3 (mutually exclusive)
  "form_issues": ["knees over toes"],   // 0-3 (multi-label)
  "variant": null,                      // null or "hold"
  "visible": true                       // false nullifies all fields
}
```

This yields 12 possible labels across 5 groups. We use **generative classification** (JSON output) rather than a classification head because it (1) preserves coaching generation capability, (2) enables zero-shot evaluation, and (3) supports agentic reasoning chains with tool invocation.

**Why not a classification head?** A discriminative head over the same LoRA features would likely achieve higher classification accuracy by avoiding the overhead of free-form generation (JSON syntax, parsing failures, token-level loss mismatches). However, a classification head would preclude three experimental conditions in our study: E1 (zero-shot requires generation), E4/E5b (agentic tool use requires reasoning in text), and coaching text generation. The generative approach trades classification accuracy for experimental breadth and practical extensibility. We acknowledge this trade-off: our absolute F1 numbers represent a lower bound on what the same model could achieve with a discriminative head on the classification subtask alone.

### 3.2 Dataset

#### 3.2.1 Source and Cleaning

The QEVD dataset provides 298K squat video clips. We clean 4,308 annotated samples to 4,032 (276 dropped, 527 conflict-resolved), then split 80/10/10 stratified by rarest label per sample. Zero overlap verified across splits.

**Annotation quality caveat.** The QEVD annotations were provided with the dataset; we did not re-annotate or compute inter-annotator agreement. The 527 conflict-resolved samples (cases where the original annotation contained internal contradictions, e.g., a stance labelled both "narrow" and "wide") were resolved programmatically via majority-rule heuristics. We do not have an estimate of the annotation noise floor, which places an unknown ceiling on achievable F1.

**Unlabelled data.** The full QEVD dataset contains 298K squat video clips, of which only 4,308 have annotations. The remaining ~294K are unused in this work. Semi-supervised or self-training approaches that leverage this unlabelled pool are a natural extension but were not explored due to scope constraints.

#### 3.2.2 Class Imbalance: Two-Stage Oversampling

The raw distribution is heavily skewed (shoulder-width: 61.8%, wide: 5.4%; no form issues: 60%).

**Stage 1 (per-group balancing)**: For each label group (stance, depth, form issues), we compute the ratio of the most frequent label's count to each label's count. Each sample's duplication weight equals the maximum such ratio across all its labels, capped at 20x. Whole-sample duplication preserves multi-label correlations.

**Stage 2 (form-issue balancing)**: Post-oversampling, form-issue samples are further duplicated to reach 50:50 parity with no-issue samples.

> **Figure 2**: Label distribution before and after oversampling.
>
> ![Figure 2: Label Distribution](thesis_figures/fig2_label_distribution.png)

> **Figure 2b**: Effect of oversampling on per-label distribution (%), with per-label duplication factors and within-group imbalance ratios.
>
> ![Figure 2b: Oversampling Effect](thesis_figures/fig2b_oversampling_effect.png)

| | Raw | Oversampled | Val/Test |
|--|:---:|:----------:|:--------:|
| Samples | 3,220 | 11,180 | 399 / 413 |
| Stance max:min | 11.4x | 1.7x | Natural |
| Depth max:min | 2.1x | 1.9x | Natural |
| Form vs no-form | 40:60 | 50:50 | Natural |

**Rationale**: We chose duplication over synthetic augmentation because video augmentation (rotation, cropping) can alter the ground-truth label, and text paraphrasing risks introducing label noise. Regularisation (weight decay=0.05, LoRA dropout=0.05) mitigates memorisation risk.

#### 3.2.3 Wild Domain (Reddit)

70 in-the-wild squat videos from Reddit fitness communities. Videos differ from QEVD in camera angle, lighting, background, and demographics. Only 3 of 12 active QEVD labels have wild support: shallow, back not straight, knees over toes. **Due to the small sample size and limited label coverage, wild-domain results should be treated as exploratory.**

### 3.3 Model Architecture

> **Figure 8**: Model architecture with frozen/trainable components and data flow.
>
> ![Figure 8: Model Architecture](thesis_figures/fig8_model_architecture.png)

**Qwen2.5-VL-7B-Instruct** with LoRA adaptation:

| Component | Params | Trainable | Rationale |
|-----------|--------|-----------|-----------|
| Vision tower | ~400M | Frozen | Pretrained on hundreds of millions of images; 4K squat videos cannot improve it (Section 5.8 documents the evidence) |
| Merger | ~10M | Full | Bridge layer; must adapt to task-specific alignment |
| LLM | ~7B | LoRA (~25M) | Efficient adaptation without catastrophic forgetting |

### 3.4 Training Procedure

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Learning rate | 1e-5 | Standard LoRA range; conservative to preserve zero-shot capability |
| LoRA rank/alpha | 32/32 (scaling=1.0) | Unit scaling; no amplification of gradient updates |
| Epochs | 5 | Val macro F1 still improving at epoch 5 (0.420); early stopping via best-checkpoint selection |
| Batch size | 64 global | 4/device x 4 GPUs x 4 grad_accum |
| Loss | Standard CE (Liger-fused) | Token-level focal loss does not address label imbalance (see Section 5.8) |
| Model selection | Best val macro F1 (generation-based) | We generate, parse, and score predictions every epoch |
| FPS | 4 | Squat rep = 2-5s; FPS=4 yields 8-20 frames per rep |

**Training curve** (validation, N=399):
```
Epoch 1:  F1 = 0.225   loss = 0.068
Epoch 2:  F1 = 0.362   loss = 0.046   (+0.137)
Epoch 3:  F1 = 0.391   loss = 0.043   (+0.029)
Epoch 4:  F1 = 0.402   loss = 0.044   (+0.011)
Epoch 5:  F1 = 0.420   loss = 0.044   (+0.018)  <-- selected
```

Eval loss plateaus after epoch 3 while F1 continues rising, validating F1-based model selection. We capped training at 5 epochs due to GPU compute constraints (each epoch requires ~2 hours on 4xA100s). Val F1 was still improving at epoch 5 (+0.018), suggesting further gains may be available with longer training. However, the diminishing marginal returns (epoch 2: +0.137, epoch 3: +0.029, epoch 4: +0.011, epoch 5: +0.018) suggest the remaining gains are modest.

**Reproducibility note.** All results in this paper are from a single training run per experiment condition. We do not report variance across random seeds. This is a limitation: modest differences between conditions (e.g., E4 vs E5b) could reflect run-to-run variance rather than method differences. Statistically significant comparisons (E1 vs E2, E2 vs E4) are large enough to be robust to this concern, but non-significant comparisons should be interpreted with additional caution.

### 3.5 Experiment Conditions

> **Figure 9**: Inference pipelines for each experiment group.
>
> ![Figure 9: Experiment Pipelines](thesis_figures/fig9_experiment_pipelines.png)

| ID | Name | Training Data | Inference | Tool Use | Tests |
|----|------|-------------|-----------|----------|-------|
| **E1** | Zero-shot | None | Direct VLM | None | Baseline |
| **E2** | SFT | 11,180 standard | Direct VLM | None | RQ1: SFT value |
| **E3** | Pose-only | None | Rule-based | MediaPipe | Measurement ceiling |
| **E4** | Always-tool | 11,180 standard (E2 model) | VLM + Pose + VLM | Always | RQ2: Tool value |
| **E5** | Agentic untrained | None | ReAct loop | Model decides | Baseline tool-use |
| **E5b** | Agentic trained | 5,590 standard + 5,590 agentic | ReAct loop | Model decides | RQ3: Agentic value |
| **E5c** | Agentic SFT-only | 11,180 standard (E2 model) | ReAct loop | Model decides | Agentic data necessity |

**Important design note on E2 vs E5b**: E2 trains on 11,180 standard samples; E5b trains on 5,590 standard + 5,590 agentic (size-matched at 11,180 total). E5b therefore sees **half** the standard classification examples. Differences between E2 and E5b cannot be attributed solely to the agentic framework; training data composition is confounded. A clean ablation (E5b on 11,180 agentic-only) would isolate the pipeline effect but was not conducted due to compute constraints.

**Clean comparisons available:**
- E1 vs E2: SFT effect (same inference, different training)
- E5 vs E5b vs E5c: Same inference pipeline, different training data
- E4 pass-1 vs E4 pass-2: Tool refinement effect (same model, same data)

### 3.6 Evaluation Metrics and Protocol

**Binary classification building blocks.** For each label, we count true positives (TP), false positives (FP), and false negatives (FN), then derive:

| Metric | Formula | Intuition |
|--------|---------|-----------|
| **Precision** | TP / (TP + FP) | Of everything the model flagged, how much was correct? |
| **Recall** | TP / (TP + FN) | Of everything that should have been flagged, how much did the model catch? |
| **F1** | 2 x Precision x Recall / (Precision + Recall) | Harmonic mean of precision and recall (0 to 1). Balances both; collapses if either is near zero. |

**Aggregation strategies.** Because we have 12 possible labels, we need a way to combine per-label scores:

| Metric | Definition |
|--------|-----------|
| **Micro F1** | Pools TP, FP, FN across all labels before computing F1. Dominated by high-frequency labels. |
| **Macro F1** | Computes F1 per label, then takes the unweighted mean. Gives equal weight to rare and common labels. |
| **Macro F1 (active)** | *Primary metric.* Same as macro F1, but restricted to labels with test-set support > 0 (labels that actually appear in the test data). Controlled test: 12 active labels; wild test: 3. |

**Task-specific metrics:**

| Metric | Definition |
|--------|-----------|
| **Holistic F1** | Set-level F1 over the form_issues field. Treats the predicted set of form issues as a single multi-label prediction and computes F1 against the ground-truth set. |
| **Stance / Depth Accuracy** | Fraction of samples where the predicted stance (or depth) exactly matches the ground truth. |
| **Label Coverage** | Fraction of active test-set labels that the model predicts at least once across the entire test set. |
| **Transfer Gap** | F1_controlled - F1_wild, computed over the 3 labels shared between both test sets (shallow, back not straight, knees over toes). A positive gap means the model performs better on controlled data. |

**Coaching metrics** (evaluated for models that generate coaching text: E4, E5, E5b, E5c):

| Metric | Definition |
|--------|-----------|
| **Issue Recall** | Fraction of ground-truth form issues mentioned in the coaching text. |
| **Issue Precision** | Fraction of mentioned issues that are actually present. |
| **Hallucination Rate** | 1 - issue precision (fraction of mentioned issues that are fabricated). |
| **Specificity** | How actionable and detailed the advice is (0 = generic, 2 = specific and corrective). |

**Statistical testing:**

| Metric | Definition |
|--------|-----------|
| **Bootstrap 95% CI** | Confidence intervals computed by resampling the test set 5,000 times with replacement, computing the metric on each resample, and taking the 2.5th and 97.5th percentiles. |
| **McNemar's test** | Tests whether two models disagree on the same samples at different rates. Applied to exact-match accuracy (all fields correct vs not) with Holm-Bonferroni correction across all 21 pairwise comparisons. Note: this collapses multi-label predictions into a binary outcome, which is conservative. Two models could differ systematically on specific labels while appearing equivalent under exact match (see Limitation 9 in Section 5.7). |

We report macro F1 (active), stance/depth accuracy, holistic F1, and label coverage for each experiment, with bootstrap 95% CIs (5,000 resamples). Transfer gaps are computed over the 3 shared labels between controlled and wild test sets.

---

## 4. Results

### 4.1 Main Classification Results

> **Table 2** ([table2_main_results.md](thesis_figures/table2_main_results.md)): Full results table with all metrics and CIs.

#### 4.1.1 Controlled Domain (QEVD Test, N=413)

| Experiment | Macro F1 | 95% CI | Stance Acc | Depth Acc | Holistic F1 | Coverage |
|-----------|:-------:|:------:|:---------:|:--------:|:----------:|:-------:|
| E1 zero-shot | 0.130 | [0.115, 0.145] | 0.211 | 0.230 | 0.088 | 58% |
| **E2 SFT** | **0.327** | [0.293, 0.361] | **0.608** | **0.462** | 0.235 | 83% |
| E3 pose-only | 0.246 | [0.220, 0.272] | 0.269 | 0.373 | 0.073 | 83% |
| E4 always-tool | 0.270 | [0.240, 0.301] | 0.559 | 0.235 | 0.208 | 92% |
| E5 agentic untrained | 0.075 | [0.057, 0.093] | 0.111 | 0.179 | 0.024 | 33% |
| E5b agentic trained | 0.246 | [0.219, 0.273] | 0.334 | 0.320 | **0.294** | 92% |
| E5c agentic SFT-only | 0.042 | [0.022, 0.063] | 0.053 | 0.179 | 0.046 | 50% |

> **Figure 7**: Multi-metric radar comparison across experiments.
>
> ![Figure 7: Radar Chart](thesis_figures/fig7_radar.png)

#### 4.1.2 Wild Domain (Reddit, N=70), Exploratory

| Experiment | Macro F1 (active, 3 labels) | 95% CI |
|-----------|:--------------------------:|:------:|
| E1 zero-shot | 0.150 | [0.101, 0.194] |
| E2 SFT | 0.178 | [0.111, 0.249] |
| E3 pose-only | 0.167 | [0.119, 0.208] |
| E4 always-tool | 0.275 | [0.212, 0.328] |
| E5b agentic trained | 0.289 | [0.216, 0.354] |

Note: CIs overlap substantially for most pairs. **No pairwise wild-domain comparison achieves statistical significance** (McNemar's p=1.0 for all pairs among E1, E2, E3, E4, E5b) except comparisons involving E5/E5c, which are significantly worse than all others.

### 4.2 Per-Label Analysis

> **Figure 6**: Per-label F1 heatmap across all experiments.
>
> ![Figure 6: F1 Heatmap](thesis_figures/fig6_f1_heatmap.png)

> **Figure 3**: Per-label F1, controlled (solid) vs wild (hatched).
>
> ![Figure 3: Per-Label Bar](thesis_figures/fig3_per_label_bar.png)

> **Table 3** ([table3_per_label.md](thesis_figures/table3_per_label.md)): Full per-label breakdown with geometric and holistic averages.

#### Controlled Domain

| Label | Support | E1 | E2 | E3 | E4 | E5b | Best |
|-------|:------:|:--:|:--:|:--:|:--:|:---:|------|
| **Stance** | | | | | | | |
| shoulder-width | 257 | 0.322 | **0.729** | 0.312 | 0.681 | 0.471 | E2 |
| narrow | 63 | 0.204 | 0.400 | 0.410 | **0.447** | 0.413 | E4 |
| wide | 23 | 0.139 | **0.375** | 0.012 | 0.337 | 0.165 | E2 |
| plie | 52 | 0.000 | **0.447** | 0.370 | 0.408 | 0.238 | E2 |
| **Depth** | | | | | | | |
| shallow | 90 | 0.358 | **0.487** | 0.425 | 0.416 | 0.395 | E2 |
| 90 degrees | 164 | 0.105 | **0.575** | 0.288 | 0.077 | 0.264 | E2 |
| over 90 degrees | 82 | 0.000 | 0.202 | **0.528** | 0.024 | 0.048 | E3 |
| **Form Issues** | | | | | | | |
| back not straight | 58 | 0.264 | **0.278** | 0.097 | 0.270 | 0.200 | E2 |
| knees over toes | 68 | 0.000 | 0.188 | 0.122 | 0.169 | **0.354** | E5b |
| insufficient | 57 | 0.000 | 0.239 | 0.000 | 0.185 | **0.329** | E5b |
| **Other** | | | | | | | |
| hold | 30 | 0.170 | 0.000 | **0.389** | 0.232 | 0.081 | E3 |
| not visible | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | none |

### 4.3 Tool Refinement: Pass-1 vs Pass-2 Analysis

A critical finding: E4 stores both pass-1 (before tool) and pass-2 (after tool refinement) predictions, allowing us to isolate the tool's effect:

**E4 pass-1 macro F1 = 0.327**, identical to E2. This is expected since E4 uses the E2 model for pass-1.

| Outcome | Count | % |
|---------|:-----:|:-:|
| Tool HELPED (pass-2 sample F1 > pass-1) | 53 | 12.8% |
| Tool HURT (pass-2 sample F1 < pass-1) | 144 | **34.9%** |
| Tool NEUTRAL (no change) | 216 | 52.3% |

**The tool refinement degrades 34.9% of predictions while helping only 12.8%.** The net effect is a drop from 0.327 (pass-1) to 0.270 (pass-2). The per-label damage is concentrated in:
- **90 degrees**: F1 drops from 0.575 (pass-1) to 0.077 (pass-2). The tool's depth thresholds systematically override correct visual assessments.
- **over 90 degrees**: F1 drops from 0.202 to 0.024.

### 4.4 Transfer Gap Analysis

> **Figure 4**: Per-label transfer gap by experiment, colour-coded by label group.
>
> ![Figure 4: Transfer Gap](thesis_figures/fig4_transfer_gap.png)

> **Table 4** ([table4_transfer_gap.md](thesis_figures/table4_transfer_gap.md)): Full per-label transfer gaps.

| Experiment | Ctrl F1 | Wild F1 (active) | Gap | 95% CI | Significant? |
|-----------|:------:|:----------------:|:---:|:------:|:-----------:|
| E1 zero-shot | 0.130 | 0.150 | -0.020 | [-0.065, 0.030] | No |
| **E2 SFT** | 0.327 | 0.178 | **+0.148** | [0.065, 0.221] | **Yes** |
| E3 pose-only | 0.246 | 0.167 | +0.079 | [0.031, 0.132] | Yes |
| **E4 always-tool** | 0.270 | 0.275 | **-0.005** | [-0.066, 0.064] | **No** |
| E5b agentic | 0.246 | 0.289 | -0.042 | [-0.114, 0.038] | No |

**Caveat**: The transfer gap is computed over only 3 shared labels. The "macro F1 (active)" metric averages over 12 labels in controlled but 3 in wild, so these numbers are **not directly comparable** as absolute values. The gap analysis (which uses matched labels) is more informative.

### 4.5 Statistical Significance

> **Table 7** ([table7_mcnemar.md](thesis_figures/table7_mcnemar.md)): Full pairwise McNemar's results with Holm-Bonferroni correction.

**Controlled domain (after Holm-Bonferroni correction):**

| Comparison | chi-sq | p-value | Significant? | Interpretation |
|-----------|:----:|:-------:|:-----------:|---------------|
| E1 vs E2 | 30.63 | <0.001 | **Yes** | SFT significantly better than zero-shot |
| E1 vs E3 | 12.00 | <0.001 | **Yes** | Pose rules significantly better than zero-shot |
| E2 vs E4 | 20.05 | <0.001 | **Yes** | E2 significantly better than E4 (tool hurts) |
| E2 vs E5b | 16.00 | <0.001 | **Yes** | E2 significantly better than E5b |
| E2 vs E3 | 4.06 | 0.044 | No (corrected) | Marginal; does not survive correction |
| E3 vs E4 | 0.39 | 0.532 | No | Indistinguishable |
| E4 vs E5b | 1.24 | 0.265 | No | Indistinguishable |

**Wild domain**: No statistically significant pairwise differences except involving E5/E5c (which are significantly worse than all others). All other wild comparisons show p=1.0 due to insufficient discordant pairs at N=70.

### 4.6 Agentic Calibration

> **Table 5** ([table5_calibration.md](thesis_figures/table5_calibration.md)): Tool invocation rates and field-level accuracy.

> **Figure 5**: Agentic confidence calibration analysis.
>
> ![Figure 5: Calibration](thesis_figures/fig5_calibration.png)

| Experiment | Tool Invoke Rate | Stance Acc (no-tool) | Depth Acc (no-tool) |
|-----------|:---------------:|:-------------------:|:------------------:|
| E5 untrained | 70.5% | 9.0% | 19.7% |
| E5b trained | **100%** | N/A | N/A |
| E5c SFT-only | 11.1% | 5.4% | 18.0% |

E5b invokes the tool on 100% of samples, having learned that measurements always help for this task. E5c barely uses the tool (11.1%), confirming that standard SFT training does not teach tool-use capability. E5 uses it 70.5% of the time but with poor calibration (9% stance accuracy when not using the tool).

### 4.7 Coaching Quality

> **Table 6** ([table6_coaching.md](thesis_figures/table6_coaching.md)): Coaching metrics for all tool-augmented experiments.

| Experiment | Issue Recall | Issue Precision | Hallucination Rate | Specificity |
|-----------|:-----------:|:--------------:|:-----------------:|:----------:|
| E4 always-tool (ctrl) | 0.389 | 0.190 | 0.810 | 1.97 |
| E4 always-tool (wild) | 0.721 | 0.205 | 0.795 | 2.00 |
| E5b agentic (ctrl) | 0.372 | 0.184 | 0.816 | 1.75 |
| E5b agentic (wild) | 0.586 | 0.099 | 0.901 | 1.97 |
| E5c SFT-only (ctrl) | 0.198 | **0.887** | **0.113** | 0.26 |

Key observations:
- **High hallucination rates** (80-90%) in E4 and E5b: the models mention form issues that are not present, likely because the agentic training data and tool output encourage verbose coaching regardless of whether issues exist.
- **E5c has lowest hallucination** (11.3%) but also lowest recall (19.8%). It barely generates coaching content at all (specificity=0.26 means mostly generic advice).
- **Specificity scores** near 2.0 for E4 and E5b indicate coaching is actionable and specific, even when hallucinated.

---

## 5. Discussion

### 5.1 RQ1: Does SFT Improve Over Zero-Shot?

**Answer: Yes, significantly.** E2 SFT (0.327) improves over E1 zero-shot (0.130) by +0.197 (p<0.001). This is the cleanest comparison in our study: same inference architecture, same prompt, different only in whether the model was fine-tuned.

The improvement is broad-based: stance accuracy nearly triples (0.211 -> 0.608), depth accuracy doubles (0.230 -> 0.462), and form issue detection improves 2.7x (holistic F1 0.088 -> 0.235). Label coverage increases from 58% to 83%.

**Where SFT fails:**
- **hold** (F1=0.000): Never predicted. Only 398/11,180 training samples (3.6%) contain hold. This temporal feature (sustained bottom position) may require explicit temporal modelling beyond frame-level processing.
- **not visible** (F1=0.000): Only 4 test samples; insufficient to evaluate.

### 5.2 RQ2: Does Tool Augmentation Help?

**Answer: It hurts controlled accuracy but stabilises transfer.**

The E4 pass-1 vs pass-2 analysis provides clean causal evidence:
- **Pass-1 (before tool) = 0.327** (identical to E2, same model, expected)
- **Pass-2 (after tool) = 0.270** (0.057 drop, p<0.001 vs E2)
- Tool helped 12.8% of samples (per-sample F1), hurt 34.9%, neutral 52.3%

The tool's primary damage is to depth classification: 90-degree F1 drops from 0.575 to 0.077 after refinement. The tool's knee-angle thresholds appear miscalibrated for the controlled domain, systematically overriding correct visual assessments with incorrect measurements (possibly due to MediaPipe landmark noise on gym videos).

However, on wild data, E4 achieves 0.275 vs E2's 0.178 (though not statistically distinguishable at N=70). The transfer gap analysis provides stronger evidence: E4's gap CI [-0.066, 0.064] includes zero while E2's [0.065, 0.221] excludes zero. **The tool provides domain-invariant anchoring**: joint angles and stance ratios are equally valid in gyms and living rooms, even if the tool's thresholds are imperfect.

### 5.3 RQ3: Does Agentic Training Help?

**Answer: Suggestive but confounded.**

E5b vs E2 shows different strengths: E5b leads holistic F1 (0.294 vs 0.235), particularly on knees_over_toes (0.354 vs 0.188) and insufficient (0.329 vs 0.239). E2 leads overall macro F1 (0.327 vs 0.246).

**However, this comparison is confounded.** E5b trains on 5,590 standard + 5,590 agentic samples, while E2 trains on 11,180 standard. E5b's lower stance/depth performance could be due to seeing half the standard classification data, not the agentic architecture. We cannot isolate the agentic framework's contribution without a clean ablation (e.g., E5b trained on 11,180 agentic-only).

**What we CAN conclude (clean comparisons):**
1. **Agentic training data is necessary** for agentic inference (E5b=0.246 vs E5c=0.042, McNemar p=0.003 uncorrected; does not survive Holm-Bonferroni correction, but the 0.204 absolute F1 difference is practically meaningful). Standard SFT models placed in agentic pipelines fail catastrophically.
2. **Agentic training outperforms no training** in agentic pipelines (E5b=0.246 vs E5=0.075).
3. **E5b achieves the highest label coverage** (92%, tied with E4), suggesting the agentic reasoning chain encourages diverse predictions.

### 5.4 RQ4: What Drives the Transfer Gap? (Exploratory)

**Answer: Domain-specific visual reliance, with caveats about statistical power.**

The directional pattern is consistent:
- Models with no tool (E1, E2): gaps of -0.020 and +0.148
- Models with tool (E4, E5b): gaps of -0.005 and -0.042
- E2 has the only gap with CI fully above zero

This suggests tool augmentation provides domain-invariant grounding. However, **we cannot make strong causal claims** because: (1) the wild test (N=70, 3 labels) lacks statistical power for most pairwise comparisons, and (2) the macro F1 metric averages over different label sets in each domain (12 vs 3), making absolute values incomparable. The gap analysis (using matched labels) is more reliable but still limited to 3 labels.

### 5.5 Complementary Strengths and Ensemble Motivation

> The heatmap (Figure 6) makes the complementary pattern visually clear: E2's hot row for stance/depth, E5b's for form issues, E3's for measurement-amenable labels.

No single method wins everywhere:

| Category | Best | Evidence |
|----------|------|---------|
| Stance + depth | E2 | Geometric F1: 0.459 vs next-best 0.341 |
| Form issues | E5b | Holistic F1: 0.294 vs E2's 0.235 |
| Measurement-amenable (over 90, hold) | E3 | over_90=0.528, hold=0.389 (both best) |
| Domain stability | E4 | Gap = -0.005 (only model with CI firmly around zero) |
| Wild: shallow | E5b | 0.449 vs E4's 0.393, E2's 0.091 |

This motivates future work on ensemble or cascaded approaches.

### 5.6 Contextualising Absolute Performance

The best macro F1 of 0.327 warrants frank discussion. Several factors contextualise this number:

1. **Task difficulty.** This is a 12-label multi-label classification problem evaluated with macro F1, which weights all labels equally regardless of frequency. Labels like "wide" (23 test samples), "hold" (30), and "not visible" (4) have very low support, and near-zero F1 on any such label drags down the macro average. The holistic F1 (0.235-0.294), which evaluates form issue detection as a set-level prediction, may better reflect practical utility.

2. **Generative overhead.** As discussed in Section 3.1, the generative classification approach introduces JSON parsing overhead, token-level loss mismatches, and mode collapse risks that a discriminative head would avoid. Our numbers represent a lower bound on what LoRA-adapted Qwen2.5-VL could achieve on this task with an appropriate classification architecture.

3. **Annotation quality ceiling.** Without inter-annotator agreement data, we cannot estimate the noise floor of the QEVD labels. If human annotators disagree on 20-30% of samples (plausible for subjective labels like "back not straight"), then macro F1 in the 0.4-0.5 range may be approaching the ceiling.

4. **Baseline context.** The rule-based E3 (pose-only) achieves 0.246 without any learning, which suggests that a substantial portion of the task is amenable to geometric reasoning. That SFT (E2) significantly exceeds this baseline validates the VLM approach, even if absolute performance remains low. A direct comparison with a classification-head baseline on the same LoRA features (not conducted due to scope) would establish whether the gap lies in the VLM approach or the generative formulation.

5. **Practical deployment threshold.** For a coaching application, per-field accuracy may matter more than macro F1. E2's stance accuracy of 0.608 and depth accuracy of 0.462 are usable as decision-support (showing the model's assessment alongside video for human review) even if not reliable enough for fully automated coaching.

### 5.7 Coaching Safety Implications

The combination of high specificity and high hallucination (Section 4.7) is particularly concerning for deployment. A model that confidently tells a user to "keep your back straighter, I can see rounding in your lumbar spine" when their back is actually neutral could cause harm: the user may overcorrect, adopt an unnatural posture, or lose trust in the system. This is not merely an accuracy problem. It is a safety problem specific to coaching applications where incorrect advice can lead to physical harm.

The hallucination pattern likely arises because the training data (and the agentic tool output) always includes coaching text, teaching the model that coaching is always expected. A potential mitigation is to condition coaching generation on classification confidence: only generate specific advice when the model's form-issue predictions exceed a calibrated threshold. Alternatively, coaching could be restricted to detected issues only, with an explicit "no issues detected" output when the form-issue set is empty. Neither approach was tested in this work.

**We recommend against deploying the coaching component without addressing this limitation.** The classification outputs (stance, depth, form issues as structured JSON) could be presented as decision-support alongside the video, but free-text coaching at current hallucination rates poses unacceptable risk.

### 5.8 Engineering Lessons

This section documents our development journey as a practical contribution for practitioners working with VLM fine-tuning. Engineering "negative results" are underreported in the literature; we include them here because the lessons generalise beyond this specific task. **All scientific claims in this paper are based on v3.1 results with a consistent evaluation protocol.** Cross-version comparisons are provided for transparency, not as evidence.

#### The Problem: Initial SFT Made Things Worse

Our first two SFT attempts (v1, v2) produced models that performed *worse* than zero-shot:

| Version | E2 SFT (ctrl) | E1 Zero-shot (ctrl)* | Outcome |
|---------|:-------------:|:--------------------:|---------|
| v1 | 0.198 | 0.211 | SFT hurt (-6%) |
| v2 | 0.171 | 0.148 | SFT marginal (+15%) |
| **v3.1** | **0.327** | **0.130** | **SFT helped (+151%)** |

*E1 baselines differ across versions because the evaluation protocol changed (FPS, resolution). Cross-version E1 values are not directly comparable.

#### Root Causes Identified

Through systematic debugging, we identified six implementation issues. **We cannot attribute the v2->v3.1 improvement to any single fix** because all six were applied simultaneously. We document them as lessons, not as ablated contributions:

1. **Unfrozen vision tower**: Fine-tuning a ViT pretrained on hundreds of millions of images using 4K squat videos destroyed generalised visual features. Freezing it preserved the model's visual understanding.

2. **Learning rate too high**: LR=5e-5 with LoRA scaling=2.0 gave effective LR=1e-4. The v2 training curve showed loss convergence by epoch 2, then eval loss rising continuously (classic overfitting). Reducing to 1e-5 with scaling=1.0 produced steady improvement through epoch 5.

3. **Unbalanced training data caused mode collapse**: The raw dataset is heavily skewed: shoulder-width stance appears in 61.8% of samples vs wide at 5.4%, and 60% of samples have no form issues. In v1, training on this distribution without oversampling caused the model to collapse onto majority labels. It predicted shoulder-width for nearly all stances and rarely flagged any form issues, achieving high token-level accuracy by exploiting the class prior.

   The two-stage oversampling strategy (Section 3.2.2), per-group balancing capped at 20x followed by form-issue parity, was introduced specifically to break this collapse. While we lack a v3.1 ablation without oversampling (noted in Section 5.9), the v1 mode collapse was the direct motivation for this pipeline change.

4. **Trained merger weights not loaded at eval**: The unfrozen merger produced trained weights saved to `non_lora_state_dict.bin`, but the eval pipeline only loaded LoRA adapters. This silently reverted half the adaptation.

5. **Right-padding for generation**: The custom evaluation loop used right-padding for a decoder-only model, producing degraded generation quality. Model selection (best macro F1) was based on unreliable metrics.

6. **Token-level focal loss**: We implemented focal loss (gamma=2.0) to address mode collapse, but token-level focal loss down-weights easy *tokens* (boilerplate JSON syntax), not easy *labels*. It does not address class imbalance in generative classification. We reverted to standard CE with Liger fusion for ~30% speedup.

#### Practical Takeaways

1. **Freeze what you cannot improve** with your data volume
2. **Address class imbalance before training**: skewed distributions cause mode collapse onto majority labels, which token-level loss cannot fix
3. **Generation-based eval is essential**: token-level loss hides classification collapse
4. **Check your padding**: decoder-only models need left-padding for generation
5. **Load all trained weights**, not just LoRA adapters
6. **Start with scaling=1.0** for LoRA: amplification causes instability on small datasets

### 5.9 Limitations and Threats to Validity

#### Threats to Internal Validity

1. **E2 vs E5b confound**: Training data composition differs (100% standard vs 50/50 mixed). Performance differences cannot be attributed to inference architecture alone. An ablation with matched data compositions is needed.

2. **No oversampling ablation**: We claim oversampling addresses mode collapse (based on v1 evidence), but v3.1 does not include an E2-without-oversampling condition. The contribution of oversampling vs other v3.1 fixes is unknown.

3. **Six simultaneous changes v2->v3.1**: We cannot attribute the improvement to any single fix. The engineering lessons (Section 5.8) are informed speculation based on the nature of each bug, not ablated evidence.

4. **Single training run**: All results are from one random seed per experiment. Modest differences between conditions (e.g., E3 vs E5b, both at 0.246) may reflect run-to-run variance. The statistically significant comparisons (E1 vs E2, E2 vs E4) have large enough effect sizes to be robust, but all other comparisons should be interpreted with this caveat.

#### Threats to External Validity

5. **Small test sets**: Controlled N=413, wild N=70. Per-label comparisons on labels with <30 support (wide: 23, hold: 30) have wide CIs. Wild-domain comparisons are statistically underpowered; most McNemar tests show p=1.0.

6. **Wild label coverage**: Only 3 of 12 labels evaluable on wild data. Transfer conclusions for stance and depth labels cannot be drawn.

7. **Single model architecture**: All results use Qwen2.5-VL-7B. Findings may not transfer to other VLM architectures (LLaVA-Video, InternVL) or model scales. In particular, the tool-augmentation trade-off (Section 5.2) may differ for larger models with stronger visual grounding.

8. **No inter-annotator agreement**: The QEVD labels were taken as given. Without a human-agreement baseline, we cannot estimate the annotation noise floor or the ceiling on achievable F1. Some labels (e.g., "back not straight") involve subjective judgements where expert disagreement may be substantial.

#### Methodological Limitations

9. **McNemar's test on exact match**: We apply McNemar's test to exact-match accuracy, which collapses the multi-label output into a binary outcome (all fields correct vs not). This is conservative: two models could systematically disagree on different label subsets while appearing equivalent under exact match. Per-label McNemar's tests were not conducted.

10. **Generative classification overhead**: As discussed in Section 3.1, the generative formulation introduces accuracy costs relative to a classification head. Our results characterise VLM *generation* performance, not VLM *representation* performance. A classification-head ablation would separate these factors.

#### Task-Specific Limitations

11. **hold never predicted by SFT (E2)**: Despite oversampled training data (398 samples), the model never predicts hold. This temporal feature may require architectures with explicit temporal modelling beyond per-frame ViT features.

12. **not visible unsolvable**: 4 test samples; no model achieves F1>0. Insufficient data for this class.

13. **Coaching hallucination**: 80-90% hallucination rate in E4/E5b coaching output. The model generates specific and actionable, but often incorrect, coaching advice. This is a significant deployment and safety concern (see Section 5.7).

14. **Agentic confidence priors**: E5b's training data uses v1-era pose tool accuracy estimates for confidence labels. These priors may be stale relative to v3.1 model capabilities.

15. **E5b tool invocation rate**: E5b invokes the pose tool on 100% of samples (Section 4.6). While we interpret this as the model learning that measurements are always helpful, an alternative explanation is that the agentic training data always included tool calls, so the model learned to always call tools regardless of utility (pattern memorisation rather than calibrated decision-making). Distinguishing these hypotheses would require training data with variable tool-call rates.

---

## 6. Conclusion

### 6.1 Summary of Findings

**Confirmed findings (statistically supported):**

1. **SFT significantly improves squat classification** on controlled-domain data (E1: 0.130 -> E2: 0.327, p<0.001). The improvement spans stance, depth, and form issue detection.

2. **Tool refinement significantly hurts controlled accuracy** (E2: 0.327 -> E4: 0.270, p<0.001). Pass-1 analysis shows the tool degrades 34.9% of predictions (per-sample F1) while helping only 12.8%.

3. **Agentic-specific training data is necessary** for agentic inference (E5b: 0.246 vs E5c: 0.042, p=0.003 uncorrected; does not survive Holm-Bonferroni correction, but the 0.204 absolute difference is practically unambiguous). Standard SFT models cannot perform confidence assessment or tool decisions.

4. **E2 SFT has a significant transfer gap** (0.148, CI [0.065, 0.221]). Fine-tuning introduces domain-specific biases.

**Directional evidence (suggestive, needs confirmation):**

5. **Tool-augmented models show smaller transfer gaps**: E4 (-0.005) and E5b (-0.042) have gap CIs including zero, while E2's excludes zero. This consistent pattern suggests tool measurements provide domain-invariant grounding, but the wild test is underpowered for individual pairwise confirmation.

6. **Agentic training improves form issue detection**: E5b holistic F1 (0.294) exceeds E2 (0.235), particularly on knees_over_toes and insufficient. However, this is confounded with training data composition.

7. **No single method dominates**: E2 leads stance/depth, E5b leads form issues, E3 leads measurement-amenable labels. An ensemble approach is warranted.

### 6.2 Practical Implications

The best system (E2) achieves macro F1 = 0.327, which is insufficient for fully automated coaching but viable for decision-support applications where a human reviews the model's assessment alongside the video. The coaching component's 80-90% hallucination rate (Section 5.7) is a safety concern that must be addressed before any deployment involving free-text feedback. The most deployment-ready configuration would use E2's structured classification output (stance, depth, form issues) without the coaching text, presented as suggestions rather than diagnoses.

### 6.3 Future Work

- **Classification-head baseline**: Fine-tune the same LoRA features with a discriminative head to separate representation quality from generative overhead
- **Ablation study**: E2 with frozen vs unfrozen vision (holding all else constant) to isolate the largest claimed contributor
- **Clean E5b comparison**: Train on 11,180 agentic-only to isolate framework vs data effects
- **Multi-seed evaluation**: Repeat key experiments (E2, E4, E5b) across 3-5 random seeds to quantify run-to-run variance
- **Larger wild dataset**: Scale beyond N=70 to enable confirmatory transfer analysis
- **Coaching improvement**: Address 80-90% hallucination rate, possibly via RLHF, constrained generation, or confidence-gated output
- **Inter-annotator agreement study**: Establish the human-agreement ceiling on QEVD labels
- **Ensemble/cascade**: Route stance/depth to E2, form issues to E5b, measurement labels to E3

---

## Appendix A: Figures and Tables Reference

All outputs in `thesis_figures/`:

### Figures
| Figure | PNG | PDF | Content |
|--------|-----|-----|---------|
| Fig 1 | ![](thesis_figures/fig1_macro_f1_controlled.png) | [PDF](thesis_figures/fig1_macro_f1_controlled.pdf) | Macro F1 bar chart (controlled) |
| Fig 1b | ![](thesis_figures/fig1_macro_f1_wild.png) | [PDF](thesis_figures/fig1_macro_f1_wild.pdf) | Macro F1 bar chart (wild) |
| Fig 2 | ![](thesis_figures/fig2_label_distribution.png) | [PDF](thesis_figures/fig2_label_distribution.pdf) | Label distribution |
| Fig 2b | ![](thesis_figures/fig2b_oversampling_effect.png) | [PDF](thesis_figures/fig2b_oversampling_effect.pdf) | Oversampling effect per label |
| Fig 3 | ![](thesis_figures/fig3_per_label_bar.png) | [PDF](thesis_figures/fig3_per_label_bar.pdf) | Per-label F1 comparison |
| Fig 4 | ![](thesis_figures/fig4_transfer_gap.png) | [PDF](thesis_figures/fig4_transfer_gap.pdf) | Transfer gap analysis |
| Fig 5 | ![](thesis_figures/fig5_calibration.png) | [PDF](thesis_figures/fig5_calibration.pdf) | Agentic calibration |
| Fig 6 | ![](thesis_figures/fig6_f1_heatmap.png) | [PDF](thesis_figures/fig6_f1_heatmap.pdf) | F1 heatmap |
| Fig 7 | ![](thesis_figures/fig7_radar.png) | [PDF](thesis_figures/fig7_radar.pdf) | Radar chart |
| Fig 8 | ![](thesis_figures/fig8_model_architecture.png) | [PDF](thesis_figures/fig8_model_architecture.pdf) | Model architecture |
| Fig 9 | ![](thesis_figures/fig9_experiment_pipelines.png) | [PDF](thesis_figures/fig9_experiment_pipelines.pdf) | Experiment inference pipelines |

### Tables
| Table | Markdown | LaTeX | Content |
|-------|----------|-------|---------|
| Table 2 | [MD](thesis_figures/table2_main_results.md) | [TeX](thesis_figures/table2_main_results.tex) | Main results with CIs |
| Table 3 | [MD](thesis_figures/table3_per_label.md) | [TeX](thesis_figures/table3_per_label.tex) | Per-label F1 |
| Table 4 | [MD](thesis_figures/table4_transfer_gap.md) | [TeX](thesis_figures/table4_transfer_gap.tex) | Transfer gaps |
| Table 5 | [MD](thesis_figures/table5_calibration.md) | [TeX](thesis_figures/table5_calibration.tex) | Agentic calibration |
| Table 6 | [MD](thesis_figures/table6_coaching.md) | [TeX](thesis_figures/table6_coaching.tex) | Coaching quality |
| Table 7 | [MD](thesis_figures/table7_mcnemar.md) | [TeX](thesis_figures/table7_mcnemar.tex) | McNemar's significance |
