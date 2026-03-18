"""
v2: Statistical analysis — bootstrap CIs + pairwise McNemar's tests.

Reads metrics.json and predictions.jsonl from all experiments and produces:
  - Summary table with bootstrap 95% CIs
  - Pairwise McNemar's test with Holm-Bonferroni correction
  - Effect sizes (Cohen's d) for controlled vs wild comparisons

Usage:
  python eval_statistical.py --experiments_dir eval_results/ \\
      --output_dir eval_results/statistical_summary/
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

from constants import ALL_LABELS, GEOMETRIC_LABELS, HOLISTIC_LABELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", default="eval_results")
    p.add_argument("--output_dir", default="eval_results/statistical_summary")
    return p.parse_args()


def load_predictions(pred_path: Path) -> list[dict]:
    preds = []
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
    return preds


def _get_pred_labels(p: dict) -> set:
    """Extract predicted labels, handling both eval_unified (pred_labels)
    and inference_with_pose (pred_labels_pass2) output formats."""
    return set(p.get("pred_labels", p.get("pred_labels_pass2", [])))


def _has_predictions(p: dict) -> bool:
    """Check if a prediction record has usable predictions."""
    if "error" in p and "pred_labels" not in p and "pred_labels_pass2" not in p:
        return False
    return True


def mcnemar_test(preds_a: list[dict], preds_b: list[dict]) -> dict:
    """Pairwise McNemar's test on exact-match accuracy."""
    # Align by ID
    map_a = {p["id"]: _get_pred_labels(p) for p in preds_a if _has_predictions(p)}
    map_b = {p["id"]: _get_pred_labels(p) for p in preds_b if _has_predictions(p)}
    gt_a = {p["id"]: set(p.get("true_labels", [])) for p in preds_a if "error" not in p or "true_labels" in p}

    common_ids = set(map_a.keys()) & set(map_b.keys()) & set(gt_a.keys())
    if not common_ids:
        return {"error": "no_common_samples"}

    # McNemar contingency: b = A wrong, B right; c = A right, B wrong
    b = 0  # A wrong, B right
    c = 0  # A right, B wrong
    for sid in common_ids:
        gt = gt_a[sid]
        a_correct = (map_a[sid] == gt)
        b_correct = (map_b[sid] == gt)
        if not a_correct and b_correct:
            b += 1
        elif a_correct and not b_correct:
            c += 1

    # McNemar's test (with continuity correction)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}


def mcnemar_per_label(preds_a: list[dict], preds_b: list[dict], label: str) -> dict:
    """Per-label McNemar's test: compares binary correct/incorrect for a single label."""
    map_a = {p["id"]: _get_pred_labels(p) for p in preds_a if _has_predictions(p)}
    map_b = {p["id"]: _get_pred_labels(p) for p in preds_b if _has_predictions(p)}
    gt_map = {p["id"]: set(p.get("true_labels", [])) for p in preds_a if "error" not in p or "true_labels" in p}

    common_ids = set(map_a.keys()) & set(map_b.keys()) & set(gt_map.keys())
    if not common_ids:
        return {"error": "no_common_samples"}

    b = 0  # A wrong, B right
    c = 0  # A right, B wrong
    for sid in common_ids:
        gt_has = label in gt_map[sid]
        a_has = label in map_a[sid]
        b_has = label in map_b[sid]
        a_correct = (a_has == gt_has)
        b_correct = (b_has == gt_has)
        if not a_correct and b_correct:
            b += 1
        elif a_correct and not b_correct:
            c += 1

    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni multiple comparison correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # All subsequent are also not significant

    return significant


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.experiments_dir)

    # Find all experiments with metrics
    experiments = {}
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "statistical_summary" or subdir.name == "transfer_analysis":
            continue
        # Check for domain subdirs or direct metrics
        for domain_dir in [subdir / "controlled", subdir / "wild", subdir]:
            metrics_path = domain_dir / "metrics.json"
            preds_path = domain_dir / "predictions.jsonl"
            if metrics_path.exists():
                name = subdir.name
                if domain_dir.name in ("controlled", "wild"):
                    name = f"{subdir.name}/{domain_dir.name}"
                experiments[name] = {
                    "metrics_path": metrics_path,
                    "preds_path": preds_path,
                }

    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiment conditions: {list(experiments.keys())}")

    # Load all metrics
    all_metrics = {}
    for name, paths in experiments.items():
        with open(paths["metrics_path"]) as f:
            all_metrics[name] = json.load(f)

    # Summary table with CIs
    summary_rows = []
    for name, m in sorted(all_metrics.items()):
        ci = m.get("bootstrap_ci", {})
        row = {
            "experiment": name,
            "macro_f1": m["macro"]["f1"],
            "macro_f1_ci": f"[{ci.get('macro_f1', {}).get('ci_low', 0):.3f}, {ci.get('macro_f1', {}).get('ci_high', 0):.3f}]" if ci else "",
            "micro_f1": m["micro"]["f1"],
            "subset_acc": m.get("subset_accuracy", 0),
            "hamming": m.get("hamming_loss", 0),
            "stance_acc": m.get("stance_accuracy", 0),
            "depth_acc": m.get("depth_accuracy", 0),
            "geo_f1": m.get("geometric_avg_f1", 0),
            "hol_f1": m.get("holistic_avg_f1", 0),
            "coverage": m.get("label_coverage", 0),
            "domain": m.get("domain", "unknown"),
            "n_samples": m.get("total_samples", 0),
        }
        summary_rows.append(row)

    # Save summary CSV
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save summary markdown
    with open(out_dir / "summary.md", "w") as f:
        f.write("| Experiment | Domain | N | Macro F1 | CI | Micro F1 | Subset Acc | Hamming | Stance | Depth | Geo F1 | Hol F1 | Cover |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r['experiment']} | {r['domain']} | {r['n_samples']} | "
                    f"{r['macro_f1']:.3f} | {r['macro_f1_ci']} | {r['micro_f1']:.3f} | "
                    f"{r['subset_acc']:.3f} | {r['hamming']:.3f} | {r['stance_acc']:.3f} | "
                    f"{r['depth_acc']:.3f} | {r['geo_f1']:.3f} | {r['hol_f1']:.3f} | "
                    f"{r['coverage']:.0%} |\n")

    # Pairwise McNemar's tests (within same domain)
    print("\nRunning pairwise McNemar's tests...")
    exp_names = sorted(experiments.keys())
    pairwise_results = []
    p_values_for_correction = []

    for a, b in combinations(exp_names, 2):
        # Only compare within same domain
        domain_a = all_metrics[a].get("domain", "controlled")
        domain_b = all_metrics[b].get("domain", "controlled")
        if domain_a != domain_b:
            continue

        preds_a = load_predictions(experiments[a]["preds_path"])
        preds_b = load_predictions(experiments[b]["preds_path"])

        if not preds_a or not preds_b:
            continue

        result = mcnemar_test(preds_a, preds_b)
        if "error" in result:
            continue

        result["experiment_a"] = a
        result["experiment_b"] = b
        result["domain"] = domain_a
        pairwise_results.append(result)
        p_values_for_correction.append(result["p_value"])

    # Apply Holm-Bonferroni correction
    if p_values_for_correction:
        corrected = holm_bonferroni(p_values_for_correction)
        for i, result in enumerate(pairwise_results):
            result["significant_corrected"] = corrected[i]

    # Save pairwise CSV
    with open(out_dir / "pairwise_mcnemar.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_a", "experiment_b", "domain", "b", "c",
                         "chi2", "p_value", "significant", "significant_corrected"])
        for r in pairwise_results:
            writer.writerow([
                r["experiment_a"], r["experiment_b"], r["domain"],
                r["b"], r["c"], f"{r['chi2']:.4f}", f"{r['p_value']:.6f}",
                "yes" if r["significant"] else "no",
                "yes" if r.get("significant_corrected", False) else "no",
            ])

    # Print
    print(f"\n{'='*70}")
    print(f"STATISTICAL SUMMARY")
    print(f"{'='*70}")
    for r in summary_rows:
        print(f"  {r['experiment']:<35} Macro F1={r['macro_f1']:.3f} {r['macro_f1_ci']}")

    if pairwise_results:
        print(f"\nPairwise McNemar's tests — exact match ({len(pairwise_results)} comparisons):")
        for r in pairwise_results:
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns"))
            corrected = " (HB)" if r.get("significant_corrected") else ""
            print(f"  {r['experiment_a']} vs {r['experiment_b']}: "
                  f"p={r['p_value']:.4f} {sig}{corrected}")

    # ── Per-label McNemar's tests ──────────────────────────────────────
    # More statistical power than exact-match McNemar's for multi-label problems,
    # since each label is tested independently as a binary classification.
    print(f"\nRunning per-label McNemar's tests...")
    per_label_results = []
    per_label_p_values = []

    for a, b in combinations(exp_names, 2):
        domain_a = all_metrics[a].get("domain", "controlled")
        domain_b = all_metrics[b].get("domain", "controlled")
        if domain_a != domain_b:
            continue

        preds_a = load_predictions(experiments[a]["preds_path"])
        preds_b = load_predictions(experiments[b]["preds_path"])
        if not preds_a or not preds_b:
            continue

        for label in ALL_LABELS:
            result = mcnemar_per_label(preds_a, preds_b, label)
            if "error" in result:
                continue
            if result["b"] + result["c"] == 0:
                continue  # No discordant pairs for this label
            result["experiment_a"] = a
            result["experiment_b"] = b
            result["domain"] = domain_a
            result["label"] = label
            result["group"] = ("geometric" if label in GEOMETRIC_LABELS
                              else "holistic" if label in HOLISTIC_LABELS
                              else "other")
            per_label_results.append(result)
            per_label_p_values.append(result["p_value"])

    # Apply Holm-Bonferroni across ALL per-label tests
    if per_label_p_values:
        corrected_pl = holm_bonferroni(per_label_p_values)
        for i, result in enumerate(per_label_results):
            result["significant_corrected"] = corrected_pl[i]

    # Save per-label McNemar's CSV
    with open(out_dir / "pairwise_mcnemar_per_label.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_a", "experiment_b", "domain", "label", "group",
                         "b", "c", "chi2", "p_value", "significant", "significant_corrected"])
        for r in per_label_results:
            writer.writerow([
                r["experiment_a"], r["experiment_b"], r["domain"], r["label"], r["group"],
                r["b"], r["c"], f"{r['chi2']:.4f}", f"{r['p_value']:.6f}",
                "yes" if r["significant"] else "no",
                "yes" if r.get("significant_corrected", False) else "no",
            ])

    # Print per-label significant results
    sig_per_label = [r for r in per_label_results if r.get("significant_corrected")]
    if sig_per_label:
        print(f"\nPer-label McNemar's — significant after Holm-Bonferroni ({len(sig_per_label)}/{len(per_label_results)}):")
        for r in sorted(sig_per_label, key=lambda x: x["p_value"]):
            print(f"  {r['experiment_a']} vs {r['experiment_b']} [{r['label']}]: "
                  f"p={r['p_value']:.6f}, b={r['b']}, c={r['c']}")
    else:
        print(f"\nPer-label McNemar's: no significant results after Holm-Bonferroni "
              f"({len(per_label_results)} tests)")

    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
