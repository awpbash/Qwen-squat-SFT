"""
v3: Create stratified train/val/test splits with per-group balanced oversampling.

Changes from v2:
  - New oversampling strategy: per-group balanced (default). Weights each sample
    by the inverse frequency of its labels within each group, so minority labels
    like 'wide' (5.4%) get boosted to match 'shoulder-width' (61.8%).
  - Old rarest-label bucketing kept as fallback (--strategy rarest_label).
  - CLI flags: --max_ratio (default 6.0), --strategy (per_group | rarest_label).
  - Stance imbalance: 11.4x → ~2x (was 6.1x with v2's 3x-capped rarest-label).

Input:  new_qevd_mrq_clean.json (4,032 cleaned samples)
Output: new_qevd_mrq_train.json (oversampled, preserving correlations)
        new_qevd_mrq_val.json   (natural distribution)
        new_qevd_mrq_test.json  (natural distribution)
"""

import argparse
import json
import re
import math
import random
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy

INPUT_PATH = Path("new_qevd_mrq_clean.json")
TRAIN_PATH = Path("new_qevd_mrq_train.json")
VAL_PATH = Path("new_qevd_mrq_val.json")
TEST_PATH = Path("new_qevd_mrq_test.json")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# ── Prompt ──────────────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """<video>
You are evaluating a squat video.

Step 1: Describe what you observe about the person's stance, depth, and form.
Step 2: Based on your observations, output the classification JSON.

Classification rules:
- "stance": Pick exactly one of: shoulder-width | narrow | wide | plie
- "depth": Pick exactly one of: shallow | 90 degrees | over 90 degrees
- "form_issues": List any that apply: back not straight | knees over toes | insufficient
  (empty list [] if no issues)
- "variant": "hold" if the person pauses at the bottom, otherwise null
- "visible": false ONLY if the squat cannot be seen at all. If false, set all other fields to null.

Output ONLY valid JSON in this schema:
{"stance": "...", "depth": "...", "form_issues": [...], "variant": "..."|null, "visible": true|false}
"""

# ── Label maps (shared) ────────────────────────────────────────────
from constants import STANCE_MAP, DEPTH_MAP, FORM_MAP, LABEL_GROUPS


def extract_labels(gpt_response: str) -> list[str]:
    m = re.search(r"\{.*\}", gpt_response, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("labels", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def labels_to_structured(labels: list[str]) -> dict:
    label_set = set(labels)
    if "squats - not visible" in label_set:
        return {"stance": None, "depth": None, "form_issues": None, "variant": None, "visible": False}

    stance = next((STANCE_MAP[l] for l in STANCE_MAP if l in label_set), None)
    depth = next((DEPTH_MAP[l] for l in DEPTH_MAP if l in label_set), None)
    form_issues = sorted([FORM_MAP[l] for l in FORM_MAP if l in label_set])
    variant = "hold" if "squats - hold" in label_set else None

    return {"stance": stance, "depth": depth, "form_issues": form_issues, "variant": variant, "visible": True}


def build_gpt_response(structured: dict) -> str:
    if not structured["visible"]:
        analysis = "- The squat is not visible in the video."
    else:
        lines = []
        if structured.get("stance"):
            lines.append(f"- Stance: {structured['stance']}")
        if structured.get("depth"):
            lines.append(f"- Depth: {structured['depth']}")
        if structured["form_issues"]:
            for issue in structured["form_issues"]:
                lines.append(f"- Form issue: {issue}")
        else:
            lines.append("- No obvious form issues")
        if structured["variant"]:
            lines.append(f"- Variant: {structured['variant']} squat")
        analysis = "\n".join(lines)

    return f"**Visual Analysis:**\n{analysis}\n\n**JSON Classification:**\n```json\n{json.dumps(structured)}\n```"


def reformat_sample(sample: dict, split: str) -> dict:
    new = deepcopy(sample)
    labels = extract_labels(new["conversations"][1]["value"])
    structured = labels_to_structured(labels)

    meta = new.get("metadata", {})
    new["conversations"][0]["value"] = CLASSIFICATION_PROMPT.strip()
    new["conversations"][1]["value"] = build_gpt_response(structured)
    meta["structured_labels"] = structured
    meta["original_flat_labels"] = labels
    meta["split"] = split
    new["metadata"] = meta
    return new


def stratified_split(data, labels_per_sample, train_r, val_r, seed):
    """Stratified split by most-rare label per sample."""
    random.seed(seed)

    all_labels = Counter()
    for labels in labels_per_sample:
        for l in labels:
            all_labels[l] += 1

    buckets = defaultdict(list)
    for i, labels in enumerate(labels_per_sample):
        if not labels:
            rarest = "__none__"
        else:
            rarest = min(labels, key=lambda l: all_labels.get(l, 999999))
        buckets[rarest].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for bucket_label, indices in buckets.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r)) if n > 2 else 0
        n_test = n - n_train - n_val

        if n_test < 0:
            n_val = 0
            n_test = n - n_train

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def oversample_by_rarest_label(data: list[dict], seed: int, max_ratio: float = 3.0) -> list[dict]:
    """Oversample by grouping samples by their rarest label, then boosting
    minority groups to match the majority group size.

    Key improvements over v1's per-group-per-label approach:
      - Preserves label correlations: duplicates WHOLE samples, so multi-label
        co-occurrences stay proportional to the original data.
      - Caps maximum oversampling ratio per bucket (max_ratio) to limit
        duplication and reduce memorization risk.
      - Uses rarest-label grouping (same as stratified_split), ensuring the
        most underrepresented labels get the most boost.

    Args:
        data: Training samples (already split).
        seed: Random seed for reproducibility.
        max_ratio: Maximum oversampling factor per bucket (e.g., 3.0 = at most 3x).
    """
    random.seed(seed)

    # Count all labels across dataset
    label_counts = Counter()
    for s in data:
        for l in s["metadata"].get("original_flat_labels", []):
            label_counts[l.replace("squats - ", "")] += 1

    # Group samples by their rarest label (same logic as stratified_split)
    buckets = defaultdict(list)
    for i, s in enumerate(data):
        labels = [l.replace("squats - ", "") for l in s["metadata"].get("original_flat_labels", [])]
        if not labels:
            buckets["__none__"].append(i)
        else:
            rarest = min(labels, key=lambda l: label_counts.get(l, 999999))
            buckets[rarest].append(i)

    # Find target: size of largest bucket
    max_bucket_size = max(len(v) for v in buckets.values())

    print(f"\nOversampling by rarest-label buckets (max_ratio={max_ratio}x):")
    print(f"  Largest bucket: {max_bucket_size} samples")

    # Oversample each bucket, with cap
    extras = []
    for bucket_label in sorted(buckets.keys()):
        indices = buckets[bucket_label]
        current = len(indices)
        # Target: match largest bucket, but cap at max_ratio
        target = min(max_bucket_size, int(current * max_ratio))
        needed = target - current

        if needed > 0:
            pool = indices * math.ceil(needed / len(indices))
            random.shuffle(pool)
            for idx in pool[:needed]:
                extras.append(deepcopy(data[idx]))

        ratio = (current + max(needed, 0)) / current if current > 0 else 0
        print(f"  {bucket_label}: {current} → {current + max(needed, 0)} ({ratio:.1f}x)")

    result = data + extras
    random.shuffle(result)
    print(f"\n  Total: {len(data)} original + {len(extras)} oversampled = {len(result)}")
    return result


def oversample_per_group_balanced(data: list[dict], seed: int, max_ratio: float = 6.0) -> list[dict]:
    """Per-group balanced oversampling: weight each sample by its most
    underrepresented label across all groups, then duplicate proportionally.

    Unlike rarest-label bucketing (which never targets majority labels like
    shoulder-width), this approach directly computes per-label weights within
    each group and assigns each sample the maximum weight across its labels.

    Algorithm:
      1. For each label group (stance, depth, form, variant, meta), find the
         max count within that group.
      2. For each sample, compute weight = max over its labels of
         (group_max / label_count), capped at max_ratio.
      3. Duplicate each sample floor(weight)-1 times, with probabilistic
         rounding for the fractional part.

    This preserves multi-label correlations (whole-sample duplication) while
    directly targeting within-group imbalance.
    """
    random.seed(seed)

    # Count labels per group
    label_counts = Counter()
    for s in data:
        for l in s["metadata"].get("original_flat_labels", []):
            label_counts[l.replace("squats - ", "")] += 1

    # Compute group maximums
    group_max = {}
    for group_name, group_labels in LABEL_GROUPS.items():
        counts = [label_counts.get(l, 0) for l in group_labels if label_counts.get(l, 0) > 0]
        group_max[group_name] = max(counts) if counts else 1

    # Build label → group mapping
    label_to_group = {}
    for group_name, group_labels in LABEL_GROUPS.items():
        for l in group_labels:
            label_to_group[l] = group_name

    print(f"\nPer-group balanced oversampling (max_ratio={max_ratio}x):")
    print(f"  Group maximums: { {g: v for g, v in group_max.items()} }")

    # Compute per-sample weights
    extras = []
    weight_dist = Counter()  # for logging

    for s in data:
        labels = [l.replace("squats - ", "") for l in s["metadata"].get("original_flat_labels", [])]
        if not labels:
            continue

        # Weight = max over labels of (group_max / label_count)
        weight = 1.0
        for l in labels:
            group = label_to_group.get(l)
            if group and label_counts.get(l, 0) > 0:
                w = group_max[group] / label_counts[l]
                weight = max(weight, w)

        weight = min(weight, max_ratio)

        # Duplicate: floor(weight)-1 full copies + probabilistic fractional copy
        n_full = int(weight) - 1  # -1 because original already exists
        frac = weight - int(weight)

        for _ in range(n_full):
            extras.append(deepcopy(s))
        if frac > 0 and random.random() < frac:
            extras.append(deepcopy(s))

        weight_dist[f"{weight:.1f}"] += 1

    result = data + extras
    random.shuffle(result)

    # Print per-label results
    new_counts = Counter()
    for s in result:
        for l in s["metadata"].get("original_flat_labels", []):
            new_counts[l.replace("squats - ", "")] += 1

    for group_name, group_labels in LABEL_GROUPS.items():
        old_counts = {l: label_counts.get(l, 0) for l in group_labels}
        new_group = {l: new_counts.get(l, 0) for l in group_labels}
        old_max = max(old_counts.values()) if old_counts else 0
        old_min = min((v for v in old_counts.values() if v > 0), default=0)
        new_max = max(new_group.values()) if new_group else 0
        new_min = min((v for v in new_group.values() if v > 0), default=0)
        old_ratio = old_max / old_min if old_min > 0 else float("inf")
        new_ratio = new_max / new_min if new_min > 0 else float("inf")
        print(f"  [{group_name}] ratio: {old_ratio:.1f}x → {new_ratio:.1f}x")
        for l in group_labels:
            old_c = label_counts.get(l, 0)
            new_c = new_counts.get(l, 0)
            boost = new_c / old_c if old_c > 0 else 0
            print(f"    {l}: {old_c} → {new_c} ({boost:.1f}x)")

    print(f"\n  Total: {len(data)} original + {len(extras)} oversampled = {len(result)}")
    return result


def balance_form_issues(data: list[dict], seed: int, target_ratio: float = 1.0) -> list[dict]:
    """Post-oversampling step: ensure samples WITH form issues are balanced
    against samples WITHOUT form issues.

    The core problem: 60% of samples have form_issues: [], so the model
    learns to always predict empty. This function duplicates form-issue
    samples until the ratio reaches target_ratio.

    Args:
        data: Oversampled training data.
        seed: Random seed.
        target_ratio: Target ratio of has-form / no-form. 1.0 = equal count.
    """
    random.seed(seed + 1)  # different seed to avoid correlation with oversampling

    has_form = [s for s in data if s["metadata"]["structured_labels"].get("form_issues")]
    no_form = [s for s in data if not s["metadata"]["structured_labels"].get("form_issues")]

    if not has_form or not no_form:
        return data

    current_ratio = len(has_form) / len(no_form)
    print(f"\nForm-issue balancing (target_ratio={target_ratio}):")
    print(f"  Before: {len(has_form)} with form issues, {len(no_form)} without (ratio={current_ratio:.2f})")

    if current_ratio >= target_ratio:
        print(f"  Already balanced, skipping.")
        return data

    target_count = int(len(no_form) * target_ratio)
    needed = target_count - len(has_form)

    # Duplicate form-issue samples
    pool = has_form * math.ceil(needed / len(has_form))
    random.shuffle(pool)
    extras = [deepcopy(s) for s in pool[:needed]]

    result = data + extras
    random.shuffle(result)

    new_has = sum(1 for s in result if s["metadata"]["structured_labels"].get("form_issues"))
    new_no = len(result) - new_has
    print(f"  After:  {new_has} with form issues, {new_no} without (ratio={new_has/new_no:.2f})")
    print(f"  Total:  {len(data)} → {len(result)} (+{len(extras)} form-issue duplicates)")

    return result


def print_label_stats(data: list[dict], name: str):
    """Print per-label and per-group stats."""
    counts = Counter()
    for s in data:
        labels = s["metadata"].get("original_flat_labels", [])
        for l in labels:
            clean = l.replace("squats - ", "")
            counts[clean] += 1

    print(f"\n{name} ({len(data)} samples):")
    for group_name, group_labels in LABEL_GROUPS.items():
        group_counts = {l: counts.get(l, 0) for l in group_labels}
        total = sum(group_counts.values())
        max_c = max(group_counts.values()) if group_counts else 0
        min_c = min(group_counts.values()) if group_counts else 0
        ratio = max_c / min_c if min_c > 0 else float("inf")
        print(f"  [{group_name}] total={total}, max:min={max_c}:{min_c} ({ratio:.1f}x)")
        for l, c in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"    {l}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    parser.add_argument("--strategy", choices=["per_group", "rarest_label"],
                        default="per_group",
                        help="Oversampling strategy (default: per_group)")
    parser.add_argument("--max_ratio", type=float, default=10.0,
                        help="Max oversampling ratio per sample (default: 10.0)")
    parser.add_argument("--form_balance_ratio", type=float, default=1.0,
                        help="Target ratio of form-issue vs no-form-issue samples (default: 1.0 = equal)")
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} cleaned samples")
    print(f"Strategy: {args.strategy}, max_ratio: {args.max_ratio}")

    # Extract labels for stratification
    labels_per_sample = []
    for sample in data:
        labels = extract_labels(sample["conversations"][1]["value"])
        labels_per_sample.append(labels)

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(
        data, labels_per_sample, TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Reformat
    train_data = [reformat_sample(data[i], "train") for i in train_idx]
    val_data = [reformat_sample(data[i], "val") for i in val_idx]
    test_data = [reformat_sample(data[i], "test") for i in test_idx]

    # Print BEFORE oversampling
    print_label_stats(train_data, "Train (before oversampling)")

    # Oversample train set
    if args.strategy == "per_group":
        train_data = oversample_per_group_balanced(train_data, SEED, args.max_ratio)
    else:
        train_data = oversample_by_rarest_label(train_data, SEED, args.max_ratio)

    # Balance form issues vs no-form-issues
    if args.form_balance_ratio > 0:
        train_data = balance_form_issues(train_data, SEED, args.form_balance_ratio)

    # Print AFTER oversampling
    print_label_stats(train_data, "Train (after oversampling)")
    print_label_stats(val_data, "Val (natural)")
    print_label_stats(test_data, "Test (natural)")

    # Save
    with open(TRAIN_PATH, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(VAL_PATH, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(TEST_PATH, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {TRAIN_PATH}: {len(train_data)} samples")
    print(f"  {VAL_PATH}: {len(val_data)} samples")
    print(f"  {TEST_PATH}: {len(test_data)} samples")


if __name__ == "__main__":
    main()
