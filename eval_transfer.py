"""
Transfer gap analysis: compare model performance on controlled vs. wild data.

Reads metrics.json and predictions.jsonl from paired controlled/wild eval
directories. Computes per-label transfer gap (F1_controlled - F1_wild) with
bootstrap 95% CIs on the gap, and reports label coverage warnings.

Important: Reddit wild data has limited label coverage (~4 of 11 QEVD labels).
Transfer gap claims are explicitly scoped to labels present in both domains.

Usage:
  python eval_transfer.py --experiments_dir eval_results/ \\
      --output_dir eval_results/transfer_analysis/
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path

from constants import (
    STANCE_LABELS, DEPTH_LABELS, FORM_LABELS, ALL_LABELS,
    GEOMETRIC_LABELS, HOLISTIC_LABELS, TEMPORAL_LABELS,
)

MIN_SUPPORT = 5  # Minimum samples in a domain to include label in gap analysis
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 123


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", default="eval_results",
                   help="Root dir containing experiment subdirectories")
    p.add_argument("--output_dir", default="eval_results/transfer_analysis")
    return p.parse_args()


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(path: Path) -> list[dict] | None:
    """Load predictions.jsonl for bootstrap computation."""
    if not path.exists():
        return None
    preds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds


def compute_per_label_f1(y_true_list: list[set], y_pred_list: list[set]) -> dict:
    """Compute per-label F1 from label sets."""
    per_label = {}
    for label in ALL_LABELS:
        tp = sum(1 for t, p in zip(y_true_list, y_pred_list) if label in t and label in p)
        fp = sum(1 for t, p in zip(y_true_list, y_pred_list) if label not in t and label in p)
        fn = sum(1 for t, p in zip(y_true_list, y_pred_list) if label in t and label not in p)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {"f1": f1, "support": tp + fn}
    # Use macro_f1_active (only labels with support > 0) to match the
    # headline metric. Raw macro_f1 over all 13 labels is misleading when
    # many labels have zero support (common in wild domain).
    active_f1s = [v["f1"] for v in per_label.values() if v["support"] > 0]
    macro_f1_active = float(np.mean(active_f1s)) if active_f1s else 0.0
    return per_label, macro_f1_active


def bootstrap_transfer_gap(ctrl_preds: list[dict], wild_preds: list[dict],
                            n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap 95% CI on macro F1 gap (controlled - wild).

    Since controlled and wild are different datasets (unpaired), we resample
    each independently on every iteration.
    """
    rng = np.random.RandomState(seed)

    # Extract label sets
    def _extract_labels(preds: list[dict]) -> tuple[list[set], list[set]]:
        y_true, y_pred = [], []
        for p in preds:
            if "error" in p and "true_labels" not in p:
                continue
            y_true.append(set(p.get("true_labels", [])))
            # Handle different prediction key names across scripts
            pred_labels = p.get("pred_labels", p.get("pred_labels_pass2", []))
            y_pred.append(set(pred_labels))
        return y_true, y_pred

    ctrl_true, ctrl_pred = _extract_labels(ctrl_preds)
    wild_true, wild_pred = _extract_labels(wild_preds)

    n_ctrl = len(ctrl_true)
    n_wild = len(wild_true)

    boot_gaps = []
    boot_ctrl_f1 = []
    boot_wild_f1 = []

    for _ in range(n_boot):
        # Independent resampling
        ci = rng.choice(n_ctrl, size=n_ctrl, replace=True)
        wi = rng.choice(n_wild, size=n_wild, replace=True)

        ct = [ctrl_true[i] for i in ci]
        cp = [ctrl_pred[i] for i in ci]
        wt = [wild_true[i] for i in wi]
        wp = [wild_pred[i] for i in wi]

        _, ctrl_macro = compute_per_label_f1(ct, cp)
        _, wild_macro = compute_per_label_f1(wt, wp)

        boot_ctrl_f1.append(ctrl_macro)
        boot_wild_f1.append(wild_macro)
        boot_gaps.append(ctrl_macro - wild_macro)

    return {
        "gap_mean": float(np.mean(boot_gaps)),
        "gap_ci_low": float(np.percentile(boot_gaps, 2.5)),
        "gap_ci_high": float(np.percentile(boot_gaps, 97.5)),
        "gap_se": float(np.std(boot_gaps)),
        "ctrl_f1_mean": float(np.mean(boot_ctrl_f1)),
        "wild_f1_mean": float(np.mean(boot_wild_f1)),
        "n_ctrl": n_ctrl,
        "n_wild": n_wild,
    }


def compute_transfer_gap(controlled: dict, wild: dict) -> dict:
    """Compute per-label and aggregate transfer gap from metrics.json."""
    gaps = {}
    for label in ALL_LABELS:
        ctrl_f1 = controlled["per_label"].get(label, {}).get("f1", 0.0)
        wild_f1 = wild["per_label"].get(label, {}).get("f1", 0.0)
        gap = ctrl_f1 - wild_f1  # Positive = worse on wild

        if label in GEOMETRIC_LABELS:
            group = "geometric"
        elif label in HOLISTIC_LABELS:
            group = "holistic"
        elif label in TEMPORAL_LABELS:
            group = "temporal"
        else:
            group = "meta"

        ctrl_support = controlled["per_label"].get(label, {}).get("TP", 0) + \
                       controlled["per_label"].get(label, {}).get("FN", 0)
        wild_support = wild["per_label"].get(label, {}).get("TP", 0) + \
                       wild["per_label"].get(label, {}).get("FN", 0)

        gaps[label] = {
            "controlled_f1": ctrl_f1,
            "wild_f1": wild_f1,
            "gap": gap,
            "group": group,
            "ctrl_support": ctrl_support,
            "wild_support": wild_support,
            "both_have_data": ctrl_support >= MIN_SUPPORT and wild_support >= MIN_SUPPORT,
        }

    # Aggregate by group (only labels with sufficient data in both domains)
    for group_name, group_set in [("geometric", GEOMETRIC_LABELS), ("holistic", HOLISTIC_LABELS)]:
        group_gaps = [gaps[l] for l in ALL_LABELS if l in group_set and gaps[l]["both_have_data"]]
        if group_gaps:
            gaps[f"_avg_{group_name}"] = {
                "controlled_f1": float(np.mean([g["controlled_f1"] for g in group_gaps])),
                "wild_f1": float(np.mean([g["wild_f1"] for g in group_gaps])),
                "gap": float(np.mean([g["gap"] for g in group_gaps])),
                "group": group_name,
                "n_labels": len(group_gaps),
            }

    # Overall (only labels with data in both domains)
    all_with_data = [gaps[l] for l in ALL_LABELS if gaps[l]["both_have_data"]]
    if all_with_data:
        gaps["_avg_overall"] = {
            "controlled_f1": float(np.mean([g["controlled_f1"] for g in all_with_data])),
            "wild_f1": float(np.mean([g["wild_f1"] for g in all_with_data])),
            "gap": float(np.mean([g["gap"] for g in all_with_data])),
            "group": "overall",
            "n_labels": len(all_with_data),
        }

    return gaps


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.experiments_dir)

    # Find experiment pairs (controlled/ and wild/ subdirectories)
    experiments = {}
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ctrl = subdir / "controlled" / "metrics.json"
        wild = subdir / "wild" / "metrics.json"
        if ctrl.exists() and wild.exists():
            experiments[subdir.name] = {
                "controlled": load_metrics(ctrl),
                "wild": load_metrics(wild),
                "controlled_preds": load_predictions(subdir / "controlled" / "predictions.jsonl"),
                "wild_preds": load_predictions(subdir / "wild" / "predictions.jsonl"),
            }

    if not experiments:
        print("No paired controlled/wild experiments found.")
        print(f"Expected structure: {exp_dir}/<experiment>/controlled/metrics.json")
        print(f"                    {exp_dir}/<experiment>/wild/metrics.json")
        return

    print(f"Found {len(experiments)} paired experiments: {list(experiments.keys())}")

    # ── Coverage analysis ──────────────────────────────────────────────
    # Check which labels have sufficient data in wild domain
    # (Use first experiment as reference since wild GT is shared)
    first_exp = next(iter(experiments.values()))
    wild_metrics = first_exp["wild"]

    labels_with_wild_data = []
    labels_missing_wild = []
    for label in ALL_LABELS:
        wild_support = wild_metrics["per_label"].get(label, {}).get("TP", 0) + \
                       wild_metrics["per_label"].get(label, {}).get("FN", 0)
        if wild_support >= MIN_SUPPORT:
            labels_with_wild_data.append(label)
        else:
            labels_missing_wild.append(label)

    coverage_pct = len(labels_with_wild_data) / len(ALL_LABELS) * 100

    print(f"\n{'='*80}")
    print(f"LABEL COVERAGE IN WILD DOMAIN")
    print(f"{'='*80}")
    print(f"Labels with wild data (support >= {MIN_SUPPORT}): {len(labels_with_wild_data)}/{len(ALL_LABELS)} ({coverage_pct:.0f}%)")
    if labels_with_wild_data:
        print(f"  Present: {labels_with_wild_data}")
    if labels_missing_wild:
        print(f"  MISSING: {labels_missing_wild}")
        print(f"  NOTE: Transfer gap analysis is scoped to the {len(labels_with_wild_data)} labels above.")
        print(f"  Claims about stance/depth transfer cannot be supported without wild annotations.")

    # ── Compute transfer gaps ──────────────────────────────────────────
    all_gaps = {}
    bootstrap_results = {}
    for exp_name, data in experiments.items():
        gaps = compute_transfer_gap(data["controlled"], data["wild"])
        all_gaps[exp_name] = gaps

        # Bootstrap CI on macro F1 gap (if predictions available)
        if data["controlled_preds"] and data["wild_preds"]:
            boot = bootstrap_transfer_gap(data["controlled_preds"], data["wild_preds"])
            bootstrap_results[exp_name] = boot

    # ── Save detailed JSON ─────────────────────────────────────────────
    output = {
        "label_coverage": {
            "labels_with_wild_data": labels_with_wild_data,
            "labels_missing_wild": labels_missing_wild,
            "coverage_pct": coverage_pct,
            "min_support_threshold": MIN_SUPPORT,
        },
        "per_experiment_gaps": all_gaps,
        "bootstrap_ci": bootstrap_results,
    }
    with open(out_dir / "transfer_gaps.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Save summary CSV ───────────────────────────────────────────────
    csv_path = out_dir / "transfer_gap_per_label.csv"
    exp_names = sorted(experiments.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label", "group", "in_both_domains"]
        for exp in exp_names:
            header.extend([f"{exp}_ctrl", f"{exp}_wild", f"{exp}_gap"])
        writer.writerow(header)

        for label in ALL_LABELS + ["_avg_geometric", "_avg_holistic", "_avg_overall"]:
            if label in GEOMETRIC_LABELS:
                group = "geometric"
            elif label in HOLISTIC_LABELS:
                group = "holistic"
            elif label.startswith("_avg"):
                group = label.replace("_avg_", "")
            else:
                group = "other"

            # Check if label has data in both domains
            in_both = label in labels_with_wild_data or label.startswith("_avg")

            row = [label, group, "yes" if in_both else "NO"]
            for exp in exp_names:
                g = all_gaps[exp].get(label, {})
                row.extend([
                    f"{g.get('controlled_f1', 0):.4f}",
                    f"{g.get('wild_f1', 0):.4f}",
                    f"{g.get('gap', 0):.4f}" if in_both else "N/A",
                ])
            writer.writerow(row)

    # ── Save aggregate summary ─────────────────────────────────────────
    summary = {}
    for exp_name, gaps in all_gaps.items():
        ctrl_m = experiments[exp_name]["controlled"]
        wild_m = experiments[exp_name]["wild"]

        # Use macro_f1_active (only labels with support>0) for fair comparison.
        # Raw macro_f1 includes zero-support labels which artificially deflate
        # wild scores and inflate the gap.
        ctrl_active = ctrl_m.get("macro_f1_active", ctrl_m["macro"]["f1"])
        wild_active = wild_m.get("macro_f1_active", wild_m["macro"]["f1"])

        entry = {
            "macro_f1_controlled": ctrl_m["macro"]["f1"],
            "macro_f1_wild": wild_m["macro"]["f1"],
            "macro_f1_gap_raw": ctrl_m["macro"]["f1"] - wild_m["macro"]["f1"],
            "macro_f1_active_controlled": ctrl_active,
            "macro_f1_active_wild": wild_active,
            "macro_f1_active_gap": ctrl_active - wild_active,
            "n_active_labels_controlled": ctrl_m.get("n_active_labels", len(ALL_LABELS)),
            "n_active_labels_wild": wild_m.get("n_active_labels", len(ALL_LABELS)),
            "geometric_gap": gaps.get("_avg_geometric", {}).get("gap"),
            "holistic_gap": gaps.get("_avg_holistic", {}).get("gap"),
            "overall_gap": gaps.get("_avg_overall", {}).get("gap"),
            "n_labels_evaluated": gaps.get("_avg_overall", {}).get("n_labels", 0),
            "n_labels_total": len(ALL_LABELS),
        }
        # Add bootstrap CI if available
        if exp_name in bootstrap_results:
            boot = bootstrap_results[exp_name]
            entry["gap_ci_95"] = [boot["gap_ci_low"], boot["gap_ci_high"]]
            entry["gap_se"] = boot["gap_se"]
        summary[exp_name] = entry

    with open(out_dir / "transfer_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Print results ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"TRANSFER GAP ANALYSIS (scoped to {len(labels_with_wild_data)}/{len(ALL_LABELS)} labels)")
    print(f"{'='*80}")

    for exp_name in exp_names:
        s = summary[exp_name]
        print(f"\n{exp_name}:")
        print(f"  Macro F1 (active): controlled={s['macro_f1_active_controlled']:.3f} "
              f"({s['n_active_labels_controlled']} labels), "
              f"wild={s['macro_f1_active_wild']:.3f} "
              f"({s['n_active_labels_wild']} labels), "
              f"gap={s['macro_f1_active_gap']:+.3f}", end="")
        if "gap_ci_95" in s:
            print(f"  95% CI [{s['gap_ci_95'][0]:+.3f}, {s['gap_ci_95'][1]:+.3f}]")
        else:
            print()
        print(f"  Macro F1 (raw):    controlled={s['macro_f1_controlled']:.3f}, "
              f"wild={s['macro_f1_wild']:.3f}, "
              f"gap={s['macro_f1_gap_raw']:+.3f} (includes zero-support labels)")
        if s["geometric_gap"] is not None:
            print(f"  Geometric gap: {s['geometric_gap']:+.3f}")
        if s["holistic_gap"] is not None:
            print(f"  Holistic gap:  {s['holistic_gap']:+.3f}")

    # Key finding: which model has smallest transfer gap?
    print(f"\n--- Transfer Gap Ranking (active F1, smaller = better transfer) ---")
    ranked = sorted(summary.items(), key=lambda x: abs(x[1]["macro_f1_active_gap"]))
    for exp_name, s in ranked:
        ci_str = ""
        if "gap_ci_95" in s:
            ci_str = f"  95% CI [{s['gap_ci_95'][0]:+.3f}, {s['gap_ci_95'][1]:+.3f}]"
        print(f"  {exp_name}: |gap| = {abs(s['macro_f1_active_gap']):.3f}{ci_str}")

    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
