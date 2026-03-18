"""
EDA plots for thesis: document data distribution and imbalance in QEVD squat dataset.

Generates 6 thesis-quality figures:
  1. Label frequency — horizontal bars of all labels from clean data
  2. Per-group imbalance — within-group bars showing imbalance ratios
  3. Co-occurrence heatmap — pairwise label co-occurrence matrix
  4. Before vs after oversampling — side-by-side bars
  5. Split distribution — grouped bars (train/val/test) normalized to %
  6. Missing annotations — stacked bars for stance and depth

Usage:
  python eda_plots.py [--output_dir eda_figures]
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from constants import (
    ALL_LABELS, LABEL_GROUPS, STANCE_LABELS, DEPTH_LABELS, FORM_LABELS,
)

# ── Paths ─────────────────────────────────────────────────────────────
CLEAN_PATH = Path("new_qevd_mrq_clean.json")
TRAIN_PATH = Path("new_qevd_mrq_train.json")
VAL_PATH = Path("new_qevd_mrq_val.json")
TEST_PATH = Path("new_qevd_mrq_test.json")

# ── Styling (consistent with thesis_tables.py) ────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# Okabe-Ito colorblind-friendly palette for label groups
GROUP_COLORS = {
    "stance": "#0072B2",    # Blue
    "depth": "#D55E00",     # Vermilion
    "form": "#009E73",      # Bluish green
    "variant": "#E69F00",   # Orange
    "meta": "#CC79A7",      # Reddish purple
}

# Map each label to its group color
LABEL_TO_GROUP = {}
for group, labels in LABEL_GROUPS.items():
    for label in labels:
        LABEL_TO_GROUP[label] = group


def get_label_color(label: str) -> str:
    group = LABEL_TO_GROUP.get(label, "meta")
    return GROUP_COLORS[group]


# ── Data loading ──────────────────────────────────────────────────────

def extract_labels_from_clean(sample: dict) -> list[str]:
    """Extract flat labels from clean data (regex on conversations[1].value)."""
    text = sample["conversations"][1]["value"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("labels", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def extract_labels_from_split(sample: dict) -> list[str]:
    """Extract flat labels from train/val/test data (metadata.original_flat_labels)."""
    return sample.get("metadata", {}).get("original_flat_labels", [])


def strip_prefix(label: str) -> str:
    return label.replace("squats - ", "")


def load_clean_labels() -> list[list[str]]:
    with open(CLEAN_PATH) as f:
        data = json.load(f)
    return [[strip_prefix(l) for l in extract_labels_from_clean(s)] for s in data]


def load_split_labels(path: Path) -> list[list[str]]:
    with open(path) as f:
        data = json.load(f)
    return [[strip_prefix(l) for l in extract_labels_from_split(s)] for s in data]


def count_labels(labels_per_sample: list[list[str]]) -> Counter:
    c = Counter()
    for labels in labels_per_sample:
        for l in labels:
            c[l] += 1
    return c


# ── Plot 1: Label Frequency ──────────────────────────────────────────

def plot_label_frequency(clean_labels: list[list[str]], output_dir: Path):
    """Horizontal bar chart of all label frequencies from clean data."""
    counts = count_labels(clean_labels)
    n_samples = len(clean_labels)

    # Order by ALL_LABELS (grouped by category)
    labels = [l for l in ALL_LABELS if l in counts]
    values = [counts[l] for l in labels]
    colors = [get_label_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Label Frequency in QEVD Dataset (n = {n_samples:,})")

    # Annotate with count and percentage
    for bar, val in zip(bars, values):
        pct = val / n_samples * 100
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)", va="center", fontsize=8)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g.capitalize())
                       for g in LABEL_GROUPS]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(output_dir / "1_label_frequency.png")
    fig.savefig(output_dir / "1_label_frequency.pdf")
    plt.close(fig)
    print("  Plot 1: Label frequency")


# ── Plot 2: Per-group Imbalance ───────────────────────────────────────

def plot_group_imbalance(clean_labels: list[list[str]], output_dir: Path):
    """Within-group bar chart showing imbalance ratios."""
    counts = count_labels(clean_labels)

    groups_to_plot = ["stance", "depth", "form"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, group in zip(axes, groups_to_plot):
        group_labels = LABEL_GROUPS[group]
        vals = [counts.get(l, 0) for l in group_labels]
        max_val = max(vals) if vals else 1
        min_val = min(v for v in vals if v > 0) if any(v > 0 for v in vals) else 1
        ratio = max_val / min_val

        bars = ax.bar(range(len(group_labels)), vals,
                      color=GROUP_COLORS[group], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"{group.capitalize()} ({ratio:.1f}x imbalance)")

        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{val:,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Within-Group Label Imbalance", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "2_group_imbalance.png")
    fig.savefig(output_dir / "2_group_imbalance.pdf")
    plt.close(fig)
    print("  Plot 2: Per-group imbalance")


# ── Plot 3: Co-occurrence Heatmap ─────────────────────────────────────

def plot_cooccurrence(clean_labels: list[list[str]], output_dir: Path):
    """Pairwise label co-occurrence matrix."""
    labels_list = [l for l in ALL_LABELS]
    n = len(labels_list)
    label_idx = {l: i for i, l in enumerate(labels_list)}
    matrix = np.zeros((n, n), dtype=int)

    for sample_labels in clean_labels:
        present = [l for l in sample_labels if l in label_idx]
        for a in present:
            for b in present:
                matrix[label_idx[a]][label_idx[b]] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels_list, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=5.5, color=color)

    # Draw group boundaries
    boundaries = []
    pos = 0
    for group in LABEL_GROUPS:
        pos += len(LABEL_GROUPS[group])
        boundaries.append(pos - 0.5)
    for b in boundaries[:-1]:
        ax.axhline(b, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(b, color="gray", linewidth=0.8, linestyle="--")

    ax.set_title("Label Co-occurrence Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Co-occurrence count")
    fig.tight_layout()
    fig.savefig(output_dir / "3_cooccurrence.png")
    fig.savefig(output_dir / "3_cooccurrence.pdf")
    plt.close(fig)
    print("  Plot 3: Co-occurrence heatmap")


# ── Plot 4: Before vs After Oversampling ──────────────────────────────

def plot_oversampling_comparison(clean_labels: list[list[str]],
                                 train_labels: list[list[str]],
                                 output_dir: Path):
    """Side-by-side bars comparing raw train vs oversampled train counts."""
    # Raw train: approximate from clean data with 80% train ratio
    # Better: use the actual split sizes
    # We count from actual train data (which IS oversampled)
    # For "before", we count unique samples only (deduplicate by counting
    # the original train split before oversampling). Since we don't have
    # the pre-oversampled file, we compute from clean * 0.8 split.
    # Actually, the train file metadata has split="train", so all are train.
    # The counts in train ARE oversampled. For "before", we use clean * 0.8.

    clean_counts = count_labels(clean_labels)
    n_clean = len(clean_labels)
    train_ratio = 0.80

    # Approximate pre-oversampling train counts
    before_counts = {l: int(clean_counts.get(l, 0) * train_ratio) for l in ALL_LABELS}
    after_counts = count_labels(train_labels)

    labels = [l for l in ALL_LABELS if before_counts.get(l, 0) > 0 or after_counts.get(l, 0) > 0]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, [before_counts.get(l, 0) for l in labels],
                   width, label="Before oversampling", color="#56B4E9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, [after_counts.get(l, 0) for l in labels],
                   width, label="After oversampling", color="#D55E00", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Label Counts: Before vs After Oversampling (Training Set)")
    ax.legend()

    # Annotate with ratio
    for i, label in enumerate(labels):
        before = before_counts.get(label, 0)
        after = after_counts.get(label, 0)
        if before > 0:
            ratio = after / before
            ax.text(x[i] + width / 2, after + 10, f"{ratio:.1f}x",
                    ha="center", va="bottom", fontsize=7, color="#D55E00")

    fig.tight_layout()
    fig.savefig(output_dir / "4_oversampling_comparison.png")
    fig.savefig(output_dir / "4_oversampling_comparison.pdf")
    plt.close(fig)
    print("  Plot 4: Before vs after oversampling")


# ── Plot 5: Split Distribution ────────────────────────────────────────

def plot_split_distribution(train_labels: list[list[str]],
                            val_labels: list[list[str]],
                            test_labels: list[list[str]],
                            output_dir: Path):
    """Grouped bars showing label distribution (%) across train/val/test splits."""
    train_counts = count_labels(train_labels)
    val_counts = count_labels(val_labels)
    test_counts = count_labels(test_labels)

    n_train = len(train_labels)
    n_val = len(val_labels)
    n_test = len(test_labels)

    labels = [l for l in ALL_LABELS if train_counts.get(l, 0) + val_counts.get(l, 0) + test_counts.get(l, 0) > 0]

    train_pct = [train_counts.get(l, 0) / n_train * 100 for l in labels]
    val_pct = [val_counts.get(l, 0) / n_val * 100 for l in labels]
    test_pct = [test_counts.get(l, 0) / n_test * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, train_pct, width, label=f"Train (n={n_train:,})",
           color="#0072B2", edgecolor="white")
    ax.bar(x, val_pct, width, label=f"Val (n={n_val:,})",
           color="#E69F00", edgecolor="white")
    ax.bar(x + width, test_pct, width, label=f"Test (n={n_test:,})",
           color="#009E73", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Percentage of split (%)")
    ax.set_title("Label Distribution Across Train/Val/Test Splits")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "5_split_distribution.png")
    fig.savefig(output_dir / "5_split_distribution.pdf")
    plt.close(fig)
    print("  Plot 5: Split distribution")


# ── Plot 6: Missing Annotations ───────────────────────────────────────

def plot_missing_annotations(clean_labels: list[list[str]], output_dir: Path):
    """Stacked bars showing annotated vs missing for each label group."""
    n = len(clean_labels)

    # Count samples that have at least one label from each group
    has_stance = sum(1 for labels in clean_labels
                     if any(l in STANCE_LABELS for l in labels))
    has_depth = sum(1 for labels in clean_labels
                    if any(l in DEPTH_LABELS for l in labels))
    has_form = sum(1 for labels in clean_labels
                   if any(l in FORM_LABELS for l in labels))
    has_not_visible = sum(1 for labels in clean_labels
                         if "not visible" in labels)

    # "Missing" means no label from that group AND not "not visible"
    visible_samples = n - has_not_visible
    missing_stance = visible_samples - has_stance
    missing_depth = visible_samples - has_depth
    missing_form = visible_samples - has_form

    groups = ["Stance", "Depth", "Form"]
    annotated = [has_stance, has_depth, has_form]
    missing = [missing_stance, missing_depth, missing_form]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(groups))
    width = 0.5

    bars1 = ax.bar(x, annotated, width, label="Annotated", color="#009E73", edgecolor="white")
    bars2 = ax.bar(x, missing, width, bottom=annotated, label="Missing", color="#CC79A7", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Annotation Completeness (n = {n:,} total, {has_not_visible} not visible)")
    ax.legend()

    # Annotate
    for i, (ann, mis) in enumerate(zip(annotated, missing)):
        if ann > 0:
            ax.text(i, ann / 2, f"{ann:,}\n({ann/visible_samples*100:.1f}%)",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if mis > 0:
            ax.text(i, ann + mis / 2, f"{mis:,}\n({mis/visible_samples*100:.1f}%)",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "6_missing_annotations.png")
    fig.savefig(output_dir / "6_missing_annotations.pdf")
    plt.close(fig)
    print("  Plot 6: Missing annotations")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA plots for thesis")
    parser.add_argument("--output_dir", default="eda_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    clean_labels = load_clean_labels()
    train_labels = load_split_labels(TRAIN_PATH)
    val_labels = load_split_labels(VAL_PATH)
    test_labels = load_split_labels(TEST_PATH)

    print(f"  Clean: {len(clean_labels)}, Train: {len(train_labels)}, "
          f"Val: {len(val_labels)}, Test: {len(test_labels)}")

    print(f"\nGenerating plots in {output_dir}/...")
    plot_label_frequency(clean_labels, output_dir)
    plot_group_imbalance(clean_labels, output_dir)
    plot_cooccurrence(clean_labels, output_dir)
    plot_oversampling_comparison(clean_labels, train_labels, output_dir)
    plot_split_distribution(train_labels, val_labels, test_labels, output_dir)
    plot_missing_annotations(clean_labels, output_dir)

    print(f"\nDone! {len(list(output_dir.glob('*.png')))} PNG + "
          f"{len(list(output_dir.glob('*.pdf')))} PDF files saved to {output_dir}/")


if __name__ == "__main__":
    main()
