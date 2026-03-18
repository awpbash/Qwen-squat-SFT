"""
Generate publication-quality architecture and pipeline diagrams for thesis.

Generates:
  - Figure 8: Model architecture diagram (Qwen2.5-VL-7B-Instruct with LoRA)
  - Figure 9: Experiment inference pipeline comparison (E1-E5)

Usage:
  python generate_diagrams.py [--output_dir thesis_figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# Colors
FROZEN_COLOR = "#4A90D9"       # Steel blue for frozen
FROZEN_LIGHT = "#B8D4F0"      # Light blue fill for frozen
TRAINABLE_COLOR = "#E8792B"   # Orange for trainable
TRAINABLE_LIGHT = "#FADCC0"   # Light orange fill for trainable
OUTER_COLOR = "#555555"       # Dark gray for outer box
DATA_COLOR = "#6AAF6A"        # Green for data/IO
DATA_LIGHT = "#D4EDDA"        # Light green fill
LORA_COLOR = "#D9534F"        # Red accent for LoRA
LORA_LIGHT = "#F8D7DA"        # Light red fill
ARROW_COLOR = "#333333"       # Dark gray arrows
BG_COLOR = "#FAFAFA"          # Slight off-white for outer box fill


def draw_rounded_box(ax, xy, width, height, label, sublabel=None,
                     facecolor="white", edgecolor="black", linewidth=1.5,
                     fontsize=10, fontweight="bold", sublabel_fontsize=8,
                     text_color="black", sublabel_color="#555555",
                     boxstyle="round,pad=0.02", zorder=2):
    """Draw a rounded rectangle with centered label and optional sublabel."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)

    if sublabel:
        # Main label slightly above center, sublabel below
        ax.text(x + width / 2, y + height * 0.58, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
        ax.text(x + width / 2, y + height * 0.30, sublabel,
                ha="center", va="center", fontsize=sublabel_fontsize,
                color=sublabel_color, zorder=zorder + 1, style="italic")
    else:
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
    return box


def draw_arrow(ax, start, end, color=ARROW_COLOR, linewidth=2.0,
               arrowstyle="-|>", mutation_scale=15, zorder=3):
    """Draw a clean arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=arrowstyle,
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_arrow_label(ax, x, y, label, fontsize=8, color="#555555"):
    """Draw a label near an arrow."""
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=color, zorder=5)


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 8: Model Architecture
# ═════════════════════════════════════════════════════════════════════════

def generate_fig8(output_dir: Path):
    """Generate model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 7.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Outer box: Qwen2.5-VL-7B-Instruct ──
    outer = FancyBboxPatch(
        (1.0, 0.5), 12.0, 5.5,
        boxstyle="round,pad=0.1",
        facecolor=BG_COLOR, edgecolor=OUTER_COLOR,
        linewidth=2.5, linestyle="-", zorder=1,
    )
    ax.add_patch(outer)
    ax.text(7.0, 6.25, "Qwen2.5-VL-7B-Instruct",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=OUTER_COLOR, zorder=2)

    # ── Vision Tower (FROZEN) ──
    vt_x, vt_y = 1.7, 1.8
    vt_w, vt_h = 2.8, 3.2

    vt_box = FancyBboxPatch(
        (vt_x, vt_y), vt_w, vt_h,
        boxstyle="round,pad=0.05",
        facecolor=FROZEN_LIGHT, edgecolor=FROZEN_COLOR,
        linewidth=2.0, zorder=2,
    )
    ax.add_patch(vt_box)

    ax.text(vt_x + vt_w / 2, vt_y + vt_h * 0.78, "Vision Tower",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=FROZEN_COLOR, zorder=3)
    ax.text(vt_x + vt_w / 2, vt_y + vt_h * 0.58, "ViT-600M",
            ha="center", va="center", fontsize=10, color="#444444", zorder=3)
    ax.text(vt_x + vt_w / 2, vt_y + vt_h * 0.40, "~400M params",
            ha="center", va="center", fontsize=9, color="#666666",
            style="italic", zorder=3)

    # Frozen badge
    frozen_badge_1 = FancyBboxPatch(
        (vt_x + vt_w * 0.18, vt_y + vt_h * 0.08), vt_w * 0.64, 0.5,
        boxstyle="round,pad=0.03",
        facecolor=FROZEN_COLOR, edgecolor="none",
        linewidth=0, zorder=3,
    )
    ax.add_patch(frozen_badge_1)
    ax.text(vt_x + vt_w / 2, vt_y + vt_h * 0.08 + 0.25, "FROZEN",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=4)

    # ── Merger (TRAINABLE) ──
    mg_x, mg_y = 5.3, 2.4
    mg_w, mg_h = 2.2, 2.0

    mg_box = FancyBboxPatch(
        (mg_x, mg_y), mg_w, mg_h,
        boxstyle="round,pad=0.05",
        facecolor=TRAINABLE_LIGHT, edgecolor=TRAINABLE_COLOR,
        linewidth=2.0, zorder=2,
    )
    ax.add_patch(mg_box)

    ax.text(mg_x + mg_w / 2, mg_y + mg_h * 0.75, "Merger",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=TRAINABLE_COLOR, zorder=3)
    ax.text(mg_x + mg_w / 2, mg_y + mg_h * 0.50, "~10M params",
            ha="center", va="center", fontsize=9, color="#666666",
            style="italic", zorder=3)

    # Trainable badge
    train_badge_1 = FancyBboxPatch(
        (mg_x + mg_w * 0.10, mg_y + mg_h * 0.08), mg_w * 0.80, 0.45,
        boxstyle="round,pad=0.03",
        facecolor=TRAINABLE_COLOR, edgecolor="none",
        linewidth=0, zorder=3,
    )
    ax.add_patch(train_badge_1)
    ax.text(mg_x + mg_w / 2, mg_y + mg_h * 0.08 + 0.225, "TRAINABLE\nLR = 5e-6",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="white", zorder=4, linespacing=1.1)

    # ── Language Model (FROZEN + LoRA) ──
    lm_x, lm_y = 8.3, 1.3
    lm_w, lm_h = 4.0, 4.2

    lm_box = FancyBboxPatch(
        (lm_x, lm_y), lm_w, lm_h,
        boxstyle="round,pad=0.05",
        facecolor=FROZEN_LIGHT, edgecolor=FROZEN_COLOR,
        linewidth=2.0, zorder=2,
    )
    ax.add_patch(lm_box)

    ax.text(lm_x + lm_w / 2, lm_y + lm_h * 0.88, "Language Model",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=FROZEN_COLOR, zorder=3)
    ax.text(lm_x + lm_w / 2, lm_y + lm_h * 0.74, "Qwen2.5-7B",
            ha="center", va="center", fontsize=10, color="#444444", zorder=3)
    ax.text(lm_x + lm_w / 2, lm_y + lm_h * 0.62, "~7B params",
            ha="center", va="center", fontsize=9, color="#666666",
            style="italic", zorder=3)

    # Frozen badge for LM base
    frozen_badge_2 = FancyBboxPatch(
        (lm_x + lm_w * 0.22, lm_y + lm_h * 0.48), lm_w * 0.56, 0.45,
        boxstyle="round,pad=0.03",
        facecolor=FROZEN_COLOR, edgecolor="none",
        linewidth=0, zorder=3,
    )
    ax.add_patch(frozen_badge_2)
    ax.text(lm_x + lm_w / 2, lm_y + lm_h * 0.48 + 0.225,
            "BASE FROZEN",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=4)

    # LoRA adapter sub-box inside LM
    lora_x = lm_x + 0.35
    lora_y = lm_y + 0.25
    lora_w = lm_w - 0.70
    lora_h = 1.4

    lora_box = FancyBboxPatch(
        (lora_x, lora_y), lora_w, lora_h,
        boxstyle="round,pad=0.04",
        facecolor=TRAINABLE_LIGHT, edgecolor=TRAINABLE_COLOR,
        linewidth=1.8, linestyle="--", zorder=3,
    )
    ax.add_patch(lora_box)

    ax.text(lora_x + lora_w / 2, lora_y + lora_h * 0.72,
            "LoRA Adapters",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=TRAINABLE_COLOR, zorder=4)
    ax.text(lora_x + lora_w / 2, lora_y + lora_h * 0.45,
            "~25M params | rank=32, alpha=32",
            ha="center", va="center", fontsize=8, color="#666666",
            style="italic", zorder=4)

    # Trainable badge for LoRA
    train_badge_2 = FancyBboxPatch(
        (lora_x + lora_w * 0.13, lora_y + lora_h * 0.05), lora_w * 0.74, 0.38,
        boxstyle="round,pad=0.03",
        facecolor=TRAINABLE_COLOR, edgecolor="none",
        linewidth=0, zorder=4,
    )
    ax.add_patch(train_badge_2)
    ax.text(lora_x + lora_w / 2, lora_y + lora_h * 0.05 + 0.19,
            "TRAINABLE | LR = 1e-5",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="white", zorder=5)

    # ── Input: Video Frames ──
    inp_x, inp_y = -0.3, 2.8
    inp_w, inp_h = 1.1, 1.2
    draw_rounded_box(ax, (inp_x, inp_y), inp_w, inp_h,
                     "Video\nFrames", facecolor=DATA_LIGHT,
                     edgecolor=DATA_COLOR, fontsize=10,
                     fontweight="bold", text_color=DATA_COLOR)

    # ── Output: JSON ──
    out_x, out_y = 13.3, 2.8
    out_w, out_h = 1.1, 1.2
    draw_rounded_box(ax, (out_x, out_y), out_w, out_h,
                     "JSON\nOutput", facecolor=DATA_LIGHT,
                     edgecolor=DATA_COLOR, fontsize=10,
                     fontweight="bold", text_color=DATA_COLOR)

    # ── Arrows ──
    # Input -> Vision Tower
    draw_arrow(ax, (inp_x + inp_w, inp_y + inp_h / 2),
               (vt_x, vt_y + vt_h / 2))

    # Vision Tower -> Merger
    draw_arrow(ax, (vt_x + vt_w, vt_y + vt_h / 2),
               (mg_x, mg_y + mg_h / 2))
    draw_arrow_label(ax, (vt_x + vt_w + mg_x) / 2, vt_y + vt_h / 2 + 0.35,
                     "Visual\ntokens", fontsize=8)

    # Merger -> Language Model
    draw_arrow(ax, (mg_x + mg_w, mg_y + mg_h / 2),
               (lm_x, lm_y + lm_h / 2))
    draw_arrow_label(ax, (mg_x + mg_w + lm_x) / 2, mg_y + mg_h / 2 + 0.35,
                     "Compressed\ntokens", fontsize=8)

    # Language Model -> Output
    draw_arrow(ax, (lm_x + lm_w, lm_y + lm_h / 2),
               (out_x, out_y + out_h / 2))

    # ── Legend ──
    legend_y = 0.0
    frozen_patch = mpatches.Patch(facecolor=FROZEN_LIGHT, edgecolor=FROZEN_COLOR,
                                  linewidth=1.5, label="Frozen")
    trainable_patch = mpatches.Patch(facecolor=TRAINABLE_LIGHT, edgecolor=TRAINABLE_COLOR,
                                     linewidth=1.5, label="Trainable")
    data_patch = mpatches.Patch(facecolor=DATA_LIGHT, edgecolor=DATA_COLOR,
                                linewidth=1.5, label="Data I/O")
    ax.legend(handles=[frozen_patch, trainable_patch, data_patch],
              loc="lower center", ncol=3, fontsize=10,
              frameon=True, fancybox=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.06))

    # ── Title ──
    ax.text(7.0, 6.8, "Figure 8: Model Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#333333")

    # Save
    fig.tight_layout(pad=0.5)
    for fmt in ("png", "pdf"):
        out_path = output_dir / f"fig8_model_architecture.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 9: Experiment Inference Pipelines
# ═════════════════════════════════════════════════════════════════════════

# Experiment group colors
EXP_COLORS = {
    "direct":     {"fill": "#DDEEFF", "edge": "#3377BB", "text": "#224488"},
    "pose":       {"fill": "#D4EDDA", "edge": "#28A745", "text": "#1B6D2F"},
    "always":     {"fill": "#FFF3CD", "edge": "#D4A017", "text": "#8B6914"},
    "agentic":    {"fill": "#F8D7DA", "edge": "#C0392B", "text": "#8B1A1A"},
    "tool":       {"fill": "#E8DAEF", "edge": "#8E44AD", "text": "#5B2C6F"},
    "decision":   {"fill": "#FDEBD0", "edge": "#E67E22", "text": "#A04000"},
}


def _pipeline_box(ax, x, y, w, h, label, sublabel=None,
                  group="direct", fontsize=9, sublabel_fontsize=7.5):
    """Draw a pipeline step box."""
    c = EXP_COLORS[group]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        facecolor=c["fill"], edgecolor=c["edge"],
        linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)

    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=c["text"], zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sublabel,
                ha="center", va="center", fontsize=sublabel_fontsize,
                color=c["text"], zorder=4, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=c["text"], zorder=4)
    return box


def _io_box(ax, x, y, w, h, label, fontsize=9):
    """Draw an I/O box (Video input or JSON output)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        facecolor="#F0F0F0", edgecolor="#777777",
        linewidth=1.3, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#444444", zorder=4)
    return box


def _harrow(ax, x_start, x_end, y, color=ARROW_COLOR, lw=1.5):
    """Draw a horizontal arrow."""
    draw_arrow(ax, (x_start, y), (x_end, y), color=color, linewidth=lw,
               mutation_scale=12)


def _varrow(ax, x, y_start, y_end, color=ARROW_COLOR, lw=1.5):
    """Draw a vertical arrow."""
    draw_arrow(ax, (x, y_start), (x, y_end), color=color, linewidth=lw,
               mutation_scale=12)


def generate_fig9(output_dir: Path):
    """Generate experiment inference pipeline comparison diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7.75, 10.2, "Figure 9: Experiment Inference Pipelines",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#333333")

    # Layout parameters
    row_heights = [8.2, 5.9, 3.4, 0.2]  # y-positions for each pipeline row
    box_h = 1.1
    box_w = 1.9
    io_w = 1.1
    gap = 0.35  # gap between arrow end and next box

    label_x = -0.3  # x position for row labels

    # ────────────────────────────────────────────────────────────────
    # Row 1: E1/E2 (Direct) - Video -> VLM -> JSON
    # ────────────────────────────────────────────────────────────────
    ry = row_heights[0]

    # Row label
    ax.text(label_x, ry + box_h / 2, "E1/E2\n(Direct)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=EXP_COLORS["direct"]["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=EXP_COLORS["direct"]["fill"],
                      edgecolor=EXP_COLORS["direct"]["edge"], linewidth=1.2))

    x = 1.5
    _io_box(ax, x, ry, io_w, box_h, "Video")
    x1_end = x + io_w

    x = x1_end + gap + 0.3
    _pipeline_box(ax, x, ry, box_w + 0.6, box_h, "VLM", "Qwen2.5-VL", group="direct")
    _harrow(ax, x1_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["direct"]["edge"])
    x2_end = x + box_w + 0.6

    x = x2_end + gap + 0.3
    _io_box(ax, x, ry, io_w + 0.1, box_h, "JSON")
    _harrow(ax, x2_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["direct"]["edge"])

    # Description
    desc_x = x + io_w + 0.1 + 0.5
    ax.text(desc_x, ry + box_h / 2, "E1: Zero-shot (no fine-tuning)\nE2: SFT (fine-tuned)",
            ha="left", va="center", fontsize=9, color="#555555",
            linespacing=1.4)

    # ────────────────────────────────────────────────────────────────
    # Row 2: E3 (Pose-only) - Video -> MediaPipe -> Rules -> JSON
    # ────────────────────────────────────────────────────────────────
    ry = row_heights[1]

    ax.text(label_x, ry + box_h / 2, "E3\n(Pose-only)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=EXP_COLORS["pose"]["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=EXP_COLORS["pose"]["fill"],
                      edgecolor=EXP_COLORS["pose"]["edge"], linewidth=1.2))

    x = 1.5
    _io_box(ax, x, ry, io_w, box_h, "Video")
    x1_end = x + io_w

    x = x1_end + gap + 0.3
    _pipeline_box(ax, x, ry, box_w, box_h, "MediaPipe", "Pose Est.", group="pose")
    _harrow(ax, x1_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["pose"]["edge"])
    x2_end = x + box_w

    x = x2_end + gap + 0.3
    _pipeline_box(ax, x, ry, box_w, box_h, "Rule Engine", "Heuristics", group="pose")
    _harrow(ax, x2_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["pose"]["edge"])
    x3_end = x + box_w

    x = x3_end + gap + 0.3
    _io_box(ax, x, ry, io_w + 0.1, box_h, "JSON")
    _harrow(ax, x3_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["pose"]["edge"])

    desc_x = x + io_w + 0.1 + 0.5
    ax.text(desc_x, ry + box_h / 2, "No VLM used;\npure geometric analysis",
            ha="left", va="center", fontsize=9, color="#555555",
            linespacing=1.4)

    # ────────────────────────────────────────────────────────────────
    # Row 3: E4 (Always-tool) - Video -> VLM pass-1 -> MediaPipe -> VLM pass-2 -> JSON
    # ────────────────────────────────────────────────────────────────
    ry = row_heights[2]

    ax.text(label_x, ry + box_h / 2, "E4\n(Always-tool)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=EXP_COLORS["always"]["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=EXP_COLORS["always"]["fill"],
                      edgecolor=EXP_COLORS["always"]["edge"], linewidth=1.2))

    x = 1.5
    _io_box(ax, x, ry, io_w, box_h, "Video")
    x1_end = x + io_w

    x = x1_end + gap + 0.15
    bw4 = 1.65
    _pipeline_box(ax, x, ry, bw4, box_h, "VLM", "Pass 1", group="always")
    _harrow(ax, x1_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["always"]["edge"])
    x2_end = x + bw4

    x = x2_end + gap + 0.15
    _pipeline_box(ax, x, ry, bw4, box_h, "MediaPipe", "Pose", group="tool")
    _harrow(ax, x2_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["always"]["edge"])
    x3_end = x + bw4

    x = x3_end + gap + 0.15
    _pipeline_box(ax, x, ry, bw4 + 0.2, box_h, "VLM", "Pass 2 (refine)", group="always")
    _harrow(ax, x3_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["always"]["edge"])
    x4_end = x + bw4 + 0.2

    x = x4_end + gap + 0.15
    _io_box(ax, x, ry, io_w + 0.1, box_h, "JSON")
    _harrow(ax, x4_end + 0.05, x - 0.05, ry + box_h / 2, color=EXP_COLORS["always"]["edge"])

    desc_x = x + io_w + 0.1 + 0.5
    ax.text(desc_x, ry + box_h / 2, "Pose tool always\ninvoked between passes",
            ha="left", va="center", fontsize=9, color="#555555",
            linespacing=1.4)

    # ────────────────────────────────────────────────────────────────
    # Row 4: E5/E5b/E5c (Agentic) with ReAct loop and conditional tool
    # ────────────────────────────────────────────────────────────────
    ry = row_heights[3]

    ax.text(label_x, ry + box_h / 2 + 0.5, "E5/E5b/E5c\n(Agentic)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=EXP_COLORS["agentic"]["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=EXP_COLORS["agentic"]["fill"],
                      edgecolor=EXP_COLORS["agentic"]["edge"], linewidth=1.2))

    x = 1.5
    _io_box(ax, x, ry + 0.5, io_w, box_h, "Video")
    x1_end = x + io_w

    # VLM with ReAct loop
    x = x1_end + gap + 0.15
    react_w = 2.3
    react_box = _pipeline_box(ax, x, ry + 0.5, react_w, box_h,
                              "VLM", "ReAct loop", group="agentic")
    _harrow(ax, x1_end + 0.05, x - 0.05, ry + 0.5 + box_h / 2,
            color=EXP_COLORS["agentic"]["edge"])
    x2_end = x + react_w

    # ReAct loop arrow (self-loop shown as curved arrow above)
    loop_cx = x + react_w / 2
    loop_top = ry + 0.5 + box_h + 0.05
    loop_arc = FancyArrowPatch(
        (loop_cx - 0.5, loop_top), (loop_cx + 0.5, loop_top),
        connectionstyle="arc3,rad=-0.6",
        arrowstyle="-|>",
        color=EXP_COLORS["agentic"]["edge"],
        linewidth=1.5, mutation_scale=10, zorder=5,
    )
    ax.add_patch(loop_arc)
    ax.text(loop_cx, loop_top + 0.55, "Reason",
            ha="center", va="center", fontsize=7.5,
            color=EXP_COLORS["agentic"]["text"], zorder=5)

    # Decision diamond
    x = x2_end + gap + 0.3
    diamond_cx = x + 0.7
    diamond_cy = ry + 0.5 + box_h / 2
    diamond_size = 0.55
    diamond = plt.Polygon(
        [(diamond_cx, diamond_cy + diamond_size),
         (diamond_cx + diamond_size * 1.1, diamond_cy),
         (diamond_cx, diamond_cy - diamond_size),
         (diamond_cx - diamond_size * 1.1, diamond_cy)],
        facecolor=EXP_COLORS["decision"]["fill"],
        edgecolor=EXP_COLORS["decision"]["edge"],
        linewidth=1.5, zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(diamond_cx, diamond_cy, "Use\ntool?",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=EXP_COLORS["decision"]["text"], zorder=4, linespacing=1.0)

    _harrow(ax, x2_end + 0.05, diamond_cx - diamond_size * 1.1 - 0.05,
            ry + 0.5 + box_h / 2, color=EXP_COLORS["agentic"]["edge"])

    # "Yes" path going down to MediaPipe, then back up
    yes_x = diamond_cx
    yes_y_start = diamond_cy - diamond_size
    tool_y = ry - 0.75
    tool_w = 1.65

    # Yes label
    ax.text(diamond_cx + 0.2, yes_y_start - 0.2, "Yes",
            ha="left", va="center", fontsize=8, fontweight="bold",
            color=EXP_COLORS["tool"]["text"], zorder=5)

    # Arrow down from diamond
    _varrow(ax, diamond_cx, yes_y_start - 0.05, tool_y + box_h + 0.05,
            color=EXP_COLORS["tool"]["edge"])

    # MediaPipe tool box below
    _pipeline_box(ax, diamond_cx - tool_w / 2, tool_y, tool_w, box_h,
                  "MediaPipe", "Pose Tool", group="tool")

    # Arrow from MediaPipe back up-right to "VLM continues" box
    # Draw as L-shaped path: right from MediaPipe, then up
    mp_right_x = diamond_cx + tool_w / 2
    cont_x = diamond_cx + diamond_size * 1.1 + gap + 0.6
    cont_w = 2.3

    # Horizontal line from MediaPipe to below the continue box
    ax.annotate("", xy=(cont_x + cont_w / 2, tool_y + box_h + 0.05),
                xytext=(mp_right_x + 0.05, tool_y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=EXP_COLORS["tool"]["edge"],
                                linewidth=1.5, connectionstyle="arc3,rad=-0.3"),
                zorder=5)

    # "No" path going right to "VLM continues"
    no_x_start = diamond_cx + diamond_size * 1.1
    ax.text(no_x_start + 0.15, diamond_cy + 0.25, "No",
            ha="left", va="center", fontsize=8, fontweight="bold",
            color=EXP_COLORS["agentic"]["text"], zorder=5)

    # VLM continues reasoning box
    _pipeline_box(ax, cont_x, ry + 0.5, cont_w, box_h,
                  "VLM", "Continue reasoning", group="agentic")
    _harrow(ax, no_x_start + 0.05, cont_x - 0.05,
            ry + 0.5 + box_h / 2, color=EXP_COLORS["agentic"]["edge"])

    cont_end = cont_x + cont_w

    # JSON output
    x = cont_end + gap + 0.3
    _io_box(ax, x, ry + 0.5, io_w + 0.1, box_h, "JSON")
    _harrow(ax, cont_end + 0.05, x - 0.05, ry + 0.5 + box_h / 2,
            color=EXP_COLORS["agentic"]["edge"])

    desc_x = x + io_w + 0.1 + 0.5
    ax.text(desc_x, ry + box_h / 2 + 0.5,
            "E5: Zero-shot agentic\nE5b: SFT + agentic data\nE5c: SFT only (no agentic)",
            ha="left", va="center", fontsize=9, color="#555555",
            linespacing=1.4)

    # ── Legend ──
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=EXP_COLORS["direct"]["fill"], edgecolor=EXP_COLORS["direct"]["edge"],
              linewidth=1.2, label="VLM (Direct)"),
        Patch(facecolor=EXP_COLORS["pose"]["fill"], edgecolor=EXP_COLORS["pose"]["edge"],
              linewidth=1.2, label="Pose / Rules"),
        Patch(facecolor=EXP_COLORS["always"]["fill"], edgecolor=EXP_COLORS["always"]["edge"],
              linewidth=1.2, label="VLM (Two-pass)"),
        Patch(facecolor=EXP_COLORS["agentic"]["fill"], edgecolor=EXP_COLORS["agentic"]["edge"],
              linewidth=1.2, label="VLM (Agentic)"),
        Patch(facecolor=EXP_COLORS["tool"]["fill"], edgecolor=EXP_COLORS["tool"]["edge"],
              linewidth=1.2, label="Tool (MediaPipe)"),
        Patch(facecolor=EXP_COLORS["decision"]["fill"], edgecolor=EXP_COLORS["decision"]["edge"],
              linewidth=1.2, label="Decision Node"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=9,
              frameon=True, fancybox=True, framealpha=0.9,
              bbox_to_anchor=(0.48, -0.06))

    # Save
    fig.tight_layout(pad=0.5)
    for fmt in ("png", "pdf"):
        out_path = output_dir / f"fig9_experiment_pipelines.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis architecture and pipeline diagrams.")
    parser.add_argument("--output_dir", type=str, default="thesis_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 8: Model Architecture...")
    generate_fig8(output_dir)

    print("Generating Figure 9: Experiment Inference Pipelines...")
    generate_fig9(output_dir)

    print("\nDone! All diagrams saved to", output_dir)


if __name__ == "__main__":
    main()
