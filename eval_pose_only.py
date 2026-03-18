"""
v2: Pose-only baseline evaluation (MediaPipe rules, no VLM).

Runs pose_tool.analyze_squat() on each video and applies rule-based
thresholds to produce label predictions. Works on both QEVD and Reddit data.

Usage:
  python eval_pose_only.py --data_path new_qevd_mrq_test.json \\
      --output_dir eval_results/E3_pose_only/controlled/

  python eval_pose_only.py --data_path reddit_eval_test.json \\
      --output_dir eval_results/E3_pose_only/wild/ --domain wild
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pose_tool import analyze_squat

# Import shared label/metric definitions from eval_unified
from eval_unified import (
    ALL_LABELS, STANCE_LABELS, DEPTH_LABELS, FORM_LABELS,
    GEOMETRIC_LABELS, HOLISTIC_LABELS, TEMPORAL_LABELS,
    compute_metrics, bootstrap_ci, get_true_labels,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--video_root", default=".")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", default="eval_results/E3_pose_only")
    p.add_argument("--domain", choices=["controlled", "wild"], default="controlled")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def pose_to_labels(result: dict) -> set:
    """Convert pose_tool output to flat label set."""
    if result is None or result.get("error"):
        return {"not visible"}

    suggested = result.get("suggested_labels", {})
    labels = set()

    stance = suggested.get("stance")
    if stance:
        labels.add(stance)

    depth = suggested.get("depth")
    if depth:
        labels.add(depth)

    form_issues = suggested.get("form_issues", [])
    for issue in form_issues:
        labels.add(issue)

    variant = suggested.get("variant")
    if variant:
        labels.add(variant)

    if suggested.get("visible") is False:
        return {"not visible"}

    return labels


def process_one(args_tuple):
    """Process a single video (for multiprocessing)."""
    sid, video_abs, fps = args_tuple
    try:
        result = analyze_squat(video_abs, fps=fps)
        pred_labels = pose_to_labels(result)
        return sid, pred_labels, result
    except Exception as e:
        return sid, set(), {"error": str(e)}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Evaluating {len(data)} samples [{args.domain}] with pose-only baseline")

    # Build work items
    work = []
    sample_map = {}
    for sample in data:
        sid = sample.get("id", "")
        video_rel = sample.get("video", "")
        video_abs = str((Path(args.video_root) / video_rel).resolve())
        if Path(video_abs).exists():
            work.append((sid, video_abs, args.fps))
            sample_map[sid] = sample
        else:
            sample_map[sid] = sample

    # Run in parallel
    results_map = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, w): w[0] for w in work}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pose analysis"):
            sid, pred_labels, raw_result = future.result()
            results_map[sid] = (pred_labels, raw_result)

    # Collect predictions in order
    y_true_list = []
    y_pred_list = []
    predictions = []

    for sample in data:
        sid = sample.get("id", "")
        true_labels = get_true_labels(sample)
        if sid in results_map:
            pred_labels, raw_result = results_map[sid]
        else:
            pred_labels = set()
            raw_result = {"error": "video_missing"}

        y_true_list.append(true_labels)
        y_pred_list.append(pred_labels)

        predictions.append({
            "id": sid,
            "video": sample.get("video", ""),
            "true_labels": sorted(true_labels),
            "pred_labels": sorted(pred_labels),
            "pose_measurements": {
                k: v for k, v in raw_result.items()
                if k not in ("suggested_labels", "per_frame", "error")
            } if not raw_result.get("error") else None,
            "domain": args.domain,
        })

    # Compute metrics
    metrics = compute_metrics(y_true_list, y_pred_list)
    metrics["domain"] = args.domain
    metrics["total_samples"] = len(y_true_list)

    print("Computing bootstrap 95% CI...")
    metrics["bootstrap_ci"] = bootstrap_ci(y_true_list, y_pred_list)

    # Save
    with open(out_dir / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = out_dir / "per_label_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "group", "TP", "FP", "FN", "precision", "recall", "f1"])
        for label in ALL_LABELS:
            m = metrics["per_label"][label]
            if label in GEOMETRIC_LABELS:
                group = "geometric"
            elif label in HOLISTIC_LABELS:
                group = "holistic"
            else:
                group = "other"
            writer.writerow([label, group, m["TP"], m["FP"], m["FN"],
                           f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])

    # Print summary
    ci = metrics["bootstrap_ci"]
    print(f"\n{'='*70}")
    print(f"POSE-ONLY RESULTS — {args.domain.upper()} ({len(y_true_list)} samples)")
    print(f"{'='*70}")
    print(f"Macro F1:        {metrics['macro']['f1']:.4f}  [{ci['macro_f1']['ci_low']:.3f}, {ci['macro_f1']['ci_high']:.3f}]")
    print(f"Geometric F1:    {metrics['geometric_avg_f1']:.4f}  [{ci['geometric_avg_f1']['ci_low']:.3f}, {ci['geometric_avg_f1']['ci_high']:.3f}]")
    print(f"Holistic F1:     {metrics['holistic_avg_f1']:.4f}  [{ci['holistic_avg_f1']['ci_low']:.3f}, {ci['holistic_avg_f1']['ci_high']:.3f}]")
    print(f"Stance Acc:      {metrics['stance_accuracy']:.4f}")
    print(f"Depth Acc:       {metrics['depth_accuracy']:.4f}")
    print(f"Coverage:        {metrics['label_coverage']:.0%}")

    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
