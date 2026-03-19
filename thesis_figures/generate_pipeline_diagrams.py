#!/usr/bin/env python3
"""Generate thesis-quality pipeline diagrams for QwenSFT experiments."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Shared styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Color palette
C = {
    "data":     "#4A90D9",   # blue
    "data_lt":  "#B3D4F7",   # light blue
    "model":    "#E67E22",   # orange
    "model_lt": "#F5CBA7",   # light orange
    "tool":     "#27AE60",   # green
    "tool_lt":  "#A9DFBF",   # light green
    "output":   "#8E44AD",   # purple
    "output_lt":"#D2B4DE",   # light purple
    "grey":     "#BDC3C7",
    "dark":     "#2C3E50",
    "white":    "#FFFFFF",
    "red":      "#E74C3C",
    "red_lt":   "#F5B7B1",
    "yellow":   "#F39C12",
    "yellow_lt":"#FCF3CF",
}


def box(ax, x, y, w, h, text, color, text_color="white", fontsize=9,
         bold=False, alpha=1.0, ha="center", va="center", zorder=3,
         linewidth=0.8, edgecolor=None):
    """Draw a rounded box with centered text."""
    ec = edgecolor or color
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=ec,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=zorder+1)
    return rect


def arrow(ax, x1, y1, x2, y2, color="#2C3E50", style="-|>", lw=1.2,
          connectionstyle="arc3,rad=0", zorder=2):
    """Draw an arrow between two points."""
    ar = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        connectionstyle=connectionstyle,
        lw=lw, zorder=zorder,
        mutation_scale=12,
    )
    ax.add_patch(ar)
    return ar


def setup_ax(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Data Pipeline — raw → clean → split → oversample → SFT datasets
# ═══════════════════════════════════════════════════════════════════════════
def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_ax(ax, (-0.5, 11.5), (-0.5, 7.5))

    # Title
    ax.text(5.5, 7.1, "Data Pipeline: From Raw QEVD to Training Sets",
            ha="center", fontsize=14, fontweight="bold", color=C["dark"])

    # ── Row 1: Raw data ──
    box(ax, 2.0, 6.0, 3.0, 0.7, "QEVD Raw Dataset\n4,308 squat videos",
        C["data"], fontsize=9, bold=True)

    arrow(ax, 3.6, 6.0, 4.8, 6.0, C["dark"])

    box(ax, 6.5, 6.0, 2.8, 0.7,
        "Label Cleaning\n527 fixed · 276 dropped",
        C["yellow"], text_color=C["dark"], fontsize=8)

    arrow(ax, 7.95, 6.0, 9.0, 6.0, C["dark"])

    box(ax, 10.0, 6.0, 1.8, 0.7, "4,032\ncleaned", C["data"], fontsize=9, bold=True)

    # ── Row 2: Stratified split ──
    arrow(ax, 10.0, 5.6, 10.0, 5.1, C["dark"])
    box(ax, 10.0, 4.6, 2.2, 0.7, "Stratified Split\n(label-preserving)",
        C["grey"], text_color=C["dark"], fontsize=8)

    # Three branches from split
    arrow(ax, 9.0, 4.25, 2.5, 3.5, C["dark"])
    arrow(ax, 10.0, 4.25, 5.5, 3.5, C["dark"])
    arrow(ax, 10.8, 4.25, 9.0, 3.5, C["dark"])

    # Train / Val / Test
    box(ax, 2.5, 3.1, 2.2, 0.7, "Train\n3,220 unique", C["data"], fontsize=9, bold=True)
    box(ax, 5.5, 3.1, 1.6, 0.7, "Val\n399", C["data_lt"], text_color=C["dark"], fontsize=9)
    box(ax, 9.0, 3.1, 1.6, 0.7, "Test\n413", C["data_lt"], text_color=C["dark"], fontsize=9)

    # ── Row 3: Oversampling ──
    arrow(ax, 2.5, 2.7, 2.5, 2.2, C["dark"])
    box(ax, 2.5, 1.7, 2.6, 0.8,
        "Per-Group Oversampling\n+ Form Balancing (50:50)\nmax_ratio = 10×",
        C["yellow"], text_color=C["dark"], fontsize=7.5)

    arrow(ax, 2.5, 1.25, 2.5, 0.75, C["dark"])

    # ── Row 4: Final training sets ──
    box(ax, 2.5, 0.25, 2.6, 0.8,
        "Standard SFT Data\n11,180 samples",
        C["model"], bold=True, fontsize=9)

    arrow(ax, 3.85, 0.25, 5.3, 0.25, C["dark"])

    box(ax, 6.7, 0.25, 2.4, 0.8,
        "Agentic Transform\n(confidence grounding)",
        C["tool"], fontsize=8)

    arrow(ax, 7.95, 0.25, 9.0, 0.25, C["dark"])

    box(ax, 10.2, 0.55, 2.0, 0.5,
        "Agentic Data\n10,836", C["tool"], fontsize=8, bold=True)
    box(ax, 10.2, -0.05, 2.0, 0.5,
        "Mixed (50/50)\n10,836", C["output"], fontsize=8, bold=True)

    # ── Annotation: which experiment uses which ──
    # E2 arrow
    ax.annotate("E2 SFT", xy=(2.5, -0.2), fontsize=8, ha="center",
                color=C["model"], fontweight="bold")
    # E5b arrow
    ax.annotate("E5b Agentic", xy=(10.2, -0.55), fontsize=8, ha="center",
                color=C["output"], fontweight="bold")

    # Side note: val/test used as-is
    ax.text(7.2, 3.1, "← used for evaluation,\n   no oversampling",
            fontsize=7, color="#7F8C8D", style="italic")

    fig.savefig("thesis_figures/fig_data_pipeline.pdf")
    fig.savefig("thesis_figures/fig_data_pipeline.png", dpi=300)
    plt.close(fig)
    print("✓ fig_data_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: E2 SFT — single-pass VLM inference
# ═══════════════════════════════════════════════════════════════════════════
def fig_e2_sft():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    setup_ax(ax, (-0.5, 11), (-0.3, 4.5))

    ax.text(5.25, 4.1, "E2: Standard SFT — Single-Pass Classification",
            ha="center", fontsize=13, fontweight="bold", color=C["dark"])

    # Input
    box(ax, 0.8, 2.5, 1.4, 1.2, "Squat\nVideo\n(4 FPS)", C["data"], fontsize=9, bold=True)

    arrow(ax, 1.55, 2.5, 2.4, 2.5, C["dark"])

    # Prompt
    box(ax, 3.5, 2.5, 1.8, 1.6,
        "Prompt\n─────────\nDescribe stance,\ndepth, form.\nOutput JSON.",
        C["data_lt"], text_color=C["dark"], fontsize=7.5, linewidth=0.5)

    arrow(ax, 4.45, 2.5, 5.1, 2.5, C["dark"])

    # Model
    box(ax, 6.3, 2.5, 2.0, 1.4,
        "Qwen2.5-VL-7B\n+ LoRA (r=32)\n─────────\nVision: frozen\nMerger: LR 5e-6\nLLM: LR 1e-5",
        C["model"], fontsize=7.5, bold=False)

    arrow(ax, 7.35, 2.5, 8.1, 2.5, C["dark"])

    # Output
    box(ax, 9.5, 2.5, 2.4, 1.6,
        'Output\n─────────\nVisual Analysis:\n  stance: wide …\n{\n  "stance": "wide",\n  "depth": "shallow", …\n}',
        C["output"], fontsize=6.5)

    # Training details below
    ax.text(5.25, 0.6,
            "Training: 11,180 samples · 5 epochs · batch 64 · CE loss (Liger) · cosine LR · DeepSpeed ZeRO-2\n"
            "Target: structured JSON with 5 fields (stance, depth, form_issues, variant, visible)",
            ha="center", fontsize=8, color="#5D6D7E",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9F9", ec="#D5D8DC", lw=0.5))

    fig.savefig("thesis_figures/fig_e2_sft.pdf")
    fig.savefig("thesis_figures/fig_e2_sft.png", dpi=300)
    plt.close(fig)
    print("✓ fig_e2_sft")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Agentic Loop (E5b) — two-pass with tool decision
# ═══════════════════════════════════════════════════════════════════════════
def fig_agentic_loop():
    fig, ax = plt.subplots(figsize=(13, 9.5))
    setup_ax(ax, (-0.5, 13), (-0.5, 9.8))

    ax.text(6.5, 9.4, "E5b: Agentic Inference — Confidence-Gated Tool Use",
            ha="center", fontsize=14, fontweight="bold", color=C["dark"])

    # ═══ PASS 1: left-to-right across the top ═══
    ax.text(0.3, 8.5, "Pass 1", fontsize=11, fontweight="bold", color=C["dark"])
    ax.text(0.3, 8.1, "Assessment\n+ Decision", fontsize=8, color="#5D6D7E")

    p1_y = 7.5

    box(ax, 1.5, p1_y, 1.6, 1.0, "Squat\nVideo\n(4 FPS)", C["data"], fontsize=9, bold=True)

    arrow(ax, 2.35, p1_y, 3.0, p1_y, C["dark"])

    box(ax, 4.2, p1_y, 2.0, 1.4,
        "Agentic Prompt\n─────────\n1. Observe video\n2. Rate confidence\n   per field (H/L)\n3. Tool decision\n4. Initial JSON",
        C["data_lt"], text_color=C["dark"], fontsize=7)

    arrow(ax, 5.25, p1_y, 5.9, p1_y, C["dark"])

    box(ax, 7.0, p1_y, 2.0, 0.9,
        "Qwen2.5-VL-7B\n+ LoRA (agentic)",
        C["model"], fontsize=8, bold=True)

    arrow(ax, 8.05, p1_y, 8.7, p1_y, C["dark"])

    # Pass 1 output
    box(ax, 10.2, p1_y, 2.8, 1.6,
        "Pass 1 Output\n──────────\nASSESSMENT: …\nCONFIDENCE:\n  stance: LOW\n  depth: LOW\n  form: HIGH …\nTOOL_DECISION:\n  INVOKE_TOOL\nJSON: {…}",
        C["output_lt"], text_color=C["dark"], fontsize=6.5, linewidth=0.5)

    # ═══ Arrow straight down from Pass 1 output to diamond ═══
    diamond_x = 10.2
    diamond_y = 5.5
    diamond_size = 0.65

    arrow(ax, diamond_x, p1_y - 0.85, diamond_x, diamond_y + diamond_size + 0.1, C["dark"], lw=1.5)

    # ═══ DECISION DIAMOND ═══
    diamond = plt.Polygon([
        [diamond_x, diamond_y + diamond_size],
        [diamond_x + diamond_size * 1.3, diamond_y],
        [diamond_x, diamond_y - diamond_size],
        [diamond_x - diamond_size * 1.3, diamond_y],
    ], closed=True, fc=C["yellow"], ec=C["dark"], lw=1.2, zorder=4)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, "Any\nLOW?", ha="center", va="center",
            fontsize=8, fontweight="bold", color=C["dark"], zorder=5)

    # ═══ PASS 2 label ═══
    ax.text(0.3, 4.8, "Pass 2", fontsize=11, fontweight="bold", color=C["dark"])
    ax.text(0.3, 4.4, "Refinement", fontsize=8, color="#5D6D7E")

    # ═══ YES branch — goes LEFT from diamond, then down to tool row ═══
    yes_turn_x = 7.0  # horizontal turn point
    yes_row_y = 3.2

    # YES label
    ax.text(8.2, 5.55, "YES", fontsize=10, fontweight="bold", color=C["tool"])

    # Diamond left edge → turn left
    arrow(ax, diamond_x - diamond_size * 1.3, diamond_y,
          yes_turn_x + 0.1, diamond_y, C["tool"], lw=1.5)
    # Turn down
    arrow(ax, yes_turn_x, diamond_y - 0.1,
          yes_turn_x, yes_row_y + 0.55, C["tool"], lw=1.5)
    # Turn left to pose tool
    arrow(ax, yes_turn_x - 0.1, yes_row_y,
          3.3, yes_row_y, C["tool"], lw=1.5)

    # Pose tool
    box(ax, 2.0, yes_row_y, 2.4, 1.2,
        "MediaPipe Pose Tool\n─────────────\nknee angle: 82°\nback dev: 8°\nstance ratio: 2.1\nknee-toe: 0.12\nhold: 0.0s",
        C["tool"], fontsize=7)

    # Pose → Refinement prompt
    arrow(ax, 2.0, yes_row_y - 0.65, 2.0, yes_row_y - 1.15, C["dark"])

    refine_y = 1.6
    box(ax, 2.0, refine_y, 2.4, 0.8,
        "Refinement Prompt\n+ measurements\n+ initial JSON",
        C["tool_lt"], text_color=C["dark"], fontsize=7)

    arrow(ax, 3.25, refine_y, 4.1, refine_y, C["dark"])

    # VLM pass 2
    box(ax, 5.2, refine_y, 1.8, 0.8, "VLM\nPass 2", C["model"], fontsize=9, bold=True)

    arrow(ax, 6.15, refine_y, 7.0, refine_y, C["dark"])

    # Final output (YES)
    box(ax, 8.5, refine_y, 2.6, 0.9,
        "Revised JSON\n+ Coaching Tips",
        C["output"], fontsize=9, bold=True)

    # ═══ NO branch — goes RIGHT from diamond, then down ═══
    ax.text(12.0, 5.55, "NO", fontsize=10, fontweight="bold", color=C["red"])

    # Diamond right edge → down to "kept as-is"
    no_out_y = 3.8
    arrow(ax, diamond_x + diamond_size * 1.3, diamond_y,
          12.0, diamond_y, C["red"], lw=1.5)
    arrow(ax, 12.0, diamond_y - 0.1,
          12.0, no_out_y + 0.4, C["red"], lw=1.5)

    box(ax, 12.0, no_out_y, 1.8, 0.7,
        "Initial JSON\n(kept as-is)",
        C["output_lt"], text_color=C["dark"], fontsize=8)

    # ═══ Annotation ═══
    ax.text(6.5, 0.2, "Tool invocation rate in training data: 96.9%",
            ha="center", fontsize=8, color="#5D6D7E", style="italic")

    fig.savefig("thesis_figures/fig_agentic_loop.pdf")
    fig.savefig("thesis_figures/fig_agentic_loop.png", dpi=300)
    plt.close(fig)
    print("✓ fig_agentic_loop")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Confidence Grounding — how tool decisions are created
# ═══════════════════════════════════════════════════════════════════════════
def fig_confidence_grounding():
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax, (-0.5, 12), (-0.5, 6.5))

    ax.text(5.75, 6.1,
            "Training Data Generation: Measurement-Grounded Confidence",
            ha="center", fontsize=13, fontweight="bold", color=C["dark"])

    # ── Left: per-field confidence assignment ──
    box(ax, 1.5, 4.5, 2.4, 0.8, "Squat Video\n+ Ground Truth Labels",
        C["data"], fontsize=8, bold=True)

    arrow(ax, 2.75, 4.5, 3.6, 4.5, C["dark"])

    box(ax, 4.8, 4.5, 2.0, 0.8,
        "MediaPipe\nPose Analysis",
        C["tool"], fontsize=8, bold=True)

    # Two branches from pose
    arrow(ax, 5.85, 4.1, 5.85, 3.5, C["dark"])

    # Compare
    box(ax, 5.85, 3.0, 3.2, 0.8,
        "For each field:\nCompare pose suggestion vs ground truth",
        C["yellow"], text_color=C["dark"], fontsize=8)

    # Two outcomes
    arrow(ax, 4.2, 2.55, 2.5, 2.0, C["dark"])
    arrow(ax, 7.5, 2.55, 9.0, 2.0, C["dark"])

    # Match → LOW confidence
    box(ax, 2.5, 1.4, 3.0, 1.0,
        "Pose ≈ Ground Truth\n→ Confidence = LOW\n(tool CAN help here)",
        C["tool_lt"], text_color=C["dark"], fontsize=8)

    # Mismatch → HIGH confidence
    box(ax, 9.0, 1.4, 3.0, 1.0,
        "Pose ≠ Ground Truth\n→ Confidence = HIGH\n(tool would MISLEAD)",
        C["red_lt"], text_color=C["dark"], fontsize=8)

    # Decision rule
    arrow(ax, 2.5, 0.85, 4.5, 0.2, C["dark"])
    arrow(ax, 9.0, 0.85, 7.0, 0.2, C["dark"])

    box(ax, 5.75, 0.1, 5.5, 0.6,
        "ANY field LOW → INVOKE_TOOL  |  ALL fields HIGH → NO_TOOL_NEEDED",
        C["dark"], fontsize=9, bold=True)

    # ── Right side: example ──
    ex_x = 10.5
    ax.text(ex_x, 5.3, "Example", fontsize=10, fontweight="bold",
            color=C["dark"], ha="center")

    example_text = (
        "Ground truth: plie, over 90°\n"
        "Pose says:    plie, 90°\n"
        "─────────────────────\n"
        "stance:  plie = plie  → LOW\n"
        "depth:   90° ≠ >90°  → HIGH\n"
        "form:    (fallback)   → HIGH\n"
        "variant: (fallback)   → LOW\n"
        "visible: (always)     → HIGH\n"
        "─────────────────────\n"
        "Decision: INVOKE_TOOL"
    )
    ax.text(ex_x, 4.3, example_text, fontsize=7, fontfamily="monospace",
            color=C["dark"], ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9F9", ec="#D5D8DC", lw=0.5))

    fig.savefig("thesis_figures/fig_confidence_grounding.pdf")
    fig.savefig("thesis_figures/fig_confidence_grounding.png", dpi=300)
    plt.close(fig)
    print("✓ fig_confidence_grounding")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Pose Measurements — what MediaPipe extracts
# ═══════════════════════════════════════════════════════════════════════════
def fig_pose_measurements():
    fig, ax = plt.subplots(figsize=(13, 7))
    setup_ax(ax, (-0.5, 13), (-0.3, 7.2))

    ax.text(6.5, 6.8, "E3: Pose-Only Pipeline — MediaPipe Measurements → Rule Thresholds",
            ha="center", fontsize=13, fontweight="bold", color=C["dark"])

    # ── Horizontal flow at top ──
    row_y = 5.5

    # Input
    box(ax, 1.2, row_y, 1.6, 1.0, "Squat\nVideo", C["data"], fontsize=9, bold=True)
    arrow(ax, 2.05, row_y, 2.7, row_y, C["dark"])

    # MediaPipe
    box(ax, 3.9, row_y, 2.0, 1.0,
        "MediaPipe\nPose Landmarker\n(33 landmarks/frame)",
        C["tool"], fontsize=8)
    arrow(ax, 4.95, row_y, 5.6, row_y, C["dark"])

    # Arrow label
    ax.text(5.3, row_y + 0.2, "extract", fontsize=7, color="#7F8C8D",
            style="italic", ha="center")

    # ── Measurement → Rule table (paired rows) ──
    # Each row: measurement on left, rule threshold on right
    table_left = 7.5   # center of measurement column
    table_right = 10.8  # center of rule column
    col_lw = 3.2       # width of each column
    col_rw = 3.8

    header_y = row_y
    box(ax, table_left, header_y, col_lw, 0.6,
        "Measurement", C["tool"], fontsize=9, bold=True)
    box(ax, table_right, header_y, col_rw, 0.6,
        "Rule Threshold → Label", C["output"], fontsize=9, bold=True)

    rows = [
        ("Knee angle\n(hip-knee-ankle, avg L+R)",
         ">110° → shallow\n85–110° → 90 degrees\n<85° → over 90 degrees"),
        ("Back deviation\n(shoulder-hip vs vertical)",
         ">20° → back not straight"),
        ("Stance ratio\n(ankle dist / hip dist)",
         "<0.8 → narrow  |  0.8–1.5 → shoulder\n1.5–2.0 → wide  |  >2.0 → plie"),
        ("Knee-over-toe offset\n(horizontal displacement)",
         ">0.08 → knees over toes"),
        ("Hold duration\n(frames within 5° of min)",
         ">1.0 s → hold variant"),
    ]

    row_heights = [0.9, 0.6, 0.8, 0.6, 0.6]
    y = header_y
    for i, ((meas, rule), rh) in enumerate(zip(rows, row_heights)):
        y -= (0.6 / 2 + rh / 2 + 0.08) if i == 0 else (row_heights[i-1] / 2 + rh / 2 + 0.08)
        box(ax, table_left, y, col_lw, rh, meas,
            C["tool_lt"], text_color=C["dark"], fontsize=7.5, linewidth=0.4)
        box(ax, table_right, y, col_rw, rh, rule,
            C["output_lt"], text_color=C["dark"], fontsize=7.5, linewidth=0.4)

    # Bottom: E3 results
    ax.text(6.5, 0.3,
            "E3 excels on measurement-amenable labels: over 90° (F1=0.528), hold (F1=0.389)\n"
            "Struggles with visually-grounded labels: form issues, nuanced stance",
            ha="center", fontsize=8, color="#5D6D7E",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9F9", ec="#D5D8DC", lw=0.5))

    fig.savefig("thesis_figures/fig_pose_measurements.pdf")
    fig.savefig("thesis_figures/fig_pose_measurements.png", dpi=300)
    plt.close(fig)
    print("✓ fig_pose_measurements")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Experiment Overview — all 7 experiments at a glance
# ═══════════════════════════════════════════════════════════════════════════
def fig_experiment_overview():
    fig, ax = plt.subplots(figsize=(14, 9))
    setup_ax(ax, (-0.5, 14), (-0.5, 9.5))

    ax.text(6.75, 9.1, "Experiment Overview: Seven Configurations",
            ha="center", fontsize=14, fontweight="bold", color=C["dark"])

    # ── Column headers ──
    hdr_y = 8.3
    ax.text(1.5, hdr_y, "Experiment", fontsize=10, fontweight="bold", color=C["dark"], ha="center")
    ax.text(4.5, hdr_y, "Training Data", fontsize=10, fontweight="bold", color=C["dark"], ha="center")
    ax.text(7.5, hdr_y, "Inference Pipeline", fontsize=10, fontweight="bold", color=C["dark"], ha="center")
    ax.text(10.5, hdr_y, "Ctrl F1", fontsize=10, fontweight="bold", color=C["dark"], ha="center")
    ax.text(12.0, hdr_y, "Wild F1", fontsize=10, fontweight="bold", color=C["dark"], ha="center")

    # Separator
    ax.plot([-0.2, 13.5], [7.9, 7.9], color=C["grey"], lw=1)

    experiments = [
        ("E1\nZero-Shot", "None\n(pretrained only)", "Video → VLM → JSON", "0.130", "0.150", C["grey"]),
        ("E2\nSFT", "Standard SFT\n11,180 samples", "Video → VLM → JSON", "0.327", "0.178", C["model"]),
        ("E3\nPose-Only", "None\n(rule-based)", "Video → MediaPipe → Rules → JSON", "0.246", "0.167", C["tool"]),
        ("E4\nAlways-Tool", "Standard SFT\n(same as E2)", "Video → VLM → Pose → VLM → JSON\n(forced 2-pass)", "0.270", "0.275", C["model"]),
        ("E5\nAgentic ZS", "None\n(pretrained only)", "Video → VLM (agentic) →\n? Pose → VLM → JSON", "0.075", "0.130", C["grey"]),
        ("E5b\nAgentic SFT", "Mixed 50/50\n10,836 samples", "Video → VLM (agentic) →\n? Pose → VLM → JSON", "0.246", "0.289", C["output"]),
        ("E5c\nAgentic\n(SFT-only)", "Standard SFT\n(in agentic pipe)", "Video → VLM (agentic) →\n? Pose → VLM → JSON", "0.042", "0.062", C["red"]),
    ]

    for i, (name, data, pipe, ctrl_f1, wild_f1, color) in enumerate(experiments):
        y = 7.3 - i * 1.05

        # Highlight best
        if name.startswith("E2"):
            ax.fill_between([-0.2, 13.5], y - 0.45, y + 0.45, color=C["model_lt"], alpha=0.3, zorder=0)
        elif name.startswith("E5b"):
            ax.fill_between([-0.2, 13.5], y - 0.45, y + 0.45, color=C["output_lt"], alpha=0.3, zorder=0)

        box(ax, 1.5, y, 2.0, 0.8, name, color, fontsize=8, bold=True)
        ax.text(4.5, y, data, ha="center", va="center", fontsize=7.5, color=C["dark"])
        ax.text(7.5, y, pipe, ha="center", va="center", fontsize=7, color=C["dark"])

        # F1 scores with bold for best
        ctrl_weight = "bold" if ctrl_f1 == "0.327" else "normal"
        wild_weight = "bold" if wild_f1 == "0.289" else "normal"
        ctrl_color = C["red"] if ctrl_f1 == "0.042" else C["dark"]
        wild_color = C["red"] if wild_f1 == "0.062" else C["dark"]

        ax.text(10.5, y, ctrl_f1, ha="center", va="center", fontsize=10,
                fontweight=ctrl_weight, color=ctrl_color)
        ax.text(12.0, y, wild_f1, ha="center", va="center", fontsize=10,
                fontweight=wild_weight, color=wild_color)

        if i < len(experiments) - 1:
            ax.plot([-0.2, 13.5], [y - 0.52, y - 0.52], color="#EAECEE", lw=0.5)

    # Legend / takeaways
    ax.text(6.75, -0.1,
            "Key: E2 = best controlled (0.327)  |  E5b = best wild (0.289), near-zero transfer gap  |  E5c confirms agentic training data is essential",
            ha="center", fontsize=8, color="#5D6D7E",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9F9", ec="#D5D8DC", lw=0.5))

    fig.savefig("thesis_figures/fig_experiment_overview.pdf")
    fig.savefig("thesis_figures/fig_experiment_overview.png", dpi=300)
    plt.close(fig)
    print("✓ fig_experiment_overview")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Label taxonomy & target format
# ═══════════════════════════════════════════════════════════════════════════
def fig_label_taxonomy():
    fig, ax = plt.subplots(figsize=(13, 6))
    setup_ax(ax, (-0.5, 13), (-0.3, 6.0))

    ax.text(6.0, 5.6, "Classification Target: Label Taxonomy & Output Format",
            ha="center", fontsize=13, fontweight="bold", color=C["dark"])

    # ── Left: taxonomy tree ──
    # Root
    root_x, root_y = 1.2, 3.5
    box(ax, root_x, root_y, 1.8, 0.6, "Squat\nAssessment", C["dark"], fontsize=8, bold=True)

    groups = [
        ("Stance", "1 of 4", 4.6, ["shoulder-width", "narrow", "wide", "plie"], C["data"]),
        ("Depth", "1 of 3", 3.6, ["shallow", "90 degrees", "over 90 degrees"], C["model"]),
        ("Form Issues", "0–3", 2.6, ["back not straight", "knees over toes", "insufficient"], C["red"]),
        ("Variant", "0–1", 1.6, ["hold"], C["tool"]),
        ("Visible", "excl.", 0.7, ["true", "false"], C["grey"]),
    ]

    for label, card, y, values, color in groups:
        # Branch line from root
        arrow(ax, 2.15, root_y - 0.1, 2.8, y, C["grey"], lw=0.8)
        # Group box
        box(ax, 3.6, y, 1.6, 0.5, f"{label}\n({card})", color, fontsize=7.5, bold=True)
        # Value pills to the right
        vx = 4.6
        for v in values:
            pw = len(v) * 0.095 + 0.3
            box(ax, vx + pw/2, y, pw, 0.35, v, color + "30",
                text_color=C["dark"], fontsize=6.5, linewidth=0.4, edgecolor=color)
            vx += pw + 0.15

    # ── Right: JSON example ──
    json_x = 11.0
    ax.text(json_x, 5.0, "Output JSON", fontsize=10, fontweight="bold",
            color=C["dark"], ha="center")

    json_text = (
        '{\n'
        '  "stance": "wide",\n'
        '  "depth": "over 90 degrees",\n'
        '  "form_issues": [\n'
        '    "knees over toes"\n'
        '  ],\n'
        '  "variant": null,\n'
        '  "visible": true\n'
        '}'
    )
    ax.text(json_x, 4.4, json_text, fontsize=7.5, fontfamily="monospace",
            color=C["dark"], ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9F9", ec=C["output"], lw=1))

    # Note
    ax.text(6.0, -0.05,
            "12 unique labels across 5 groups · multi-label (form_issues) · mutually exclusive within stance/depth",
            ha="center", fontsize=7.5, color="#7F8C8D", style="italic")

    fig.savefig("thesis_figures/fig_label_taxonomy.pdf")
    fig.savefig("thesis_figures/fig_label_taxonomy.png", dpi=300)
    plt.close(fig)
    print("✓ fig_label_taxonomy")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.makedirs("thesis_figures", exist_ok=True)
    os.chdir("/tier1/home/njunwei/projects/QwenSFT")

    fig_data_pipeline()
    fig_e2_sft()
    fig_agentic_loop()
    fig_confidence_grounding()
    fig_pose_measurements()
    fig_experiment_overview()
    fig_label_taxonomy()

    print("\n✅ All 7 figures generated in thesis_figures/")
