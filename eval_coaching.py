"""
v2: Grounded coaching evaluation.

Evaluates coaching text quality against structured ground-truth labels.
Works on predictions from any experiment that includes coaching text.

Changes from v1:
  - Evaluates across both controlled and wild domains
  - Better keyword matching with regex patterns
  - Per-label breakdown of coaching coverage and hallucination

Usage:
  python eval_coaching.py --predictions_dir eval_results/E5_agentic/controlled/ \\
      --data_path new_qevd_mrq_test.json --output_dir eval_results/E6_coaching/E5_controlled/
"""

import argparse
import json
import re
import math
from pathlib import Path
from collections import Counter

# ── Label-to-coaching keyword mapping ───────────────────────────────
LABEL_KEYWORDS = {
    "shallow": [
        r"shallow", r"not deep enough", r"above parallel", r"more depth",
        r"sit lower", r"deeper", r"half squat", r"quarter squat",
        r"hip crease.*above", r"thighs.*above",
    ],
    "over 90 degrees": [
        r"deep", r"below parallel", r"full range", r"ass.to.grass",
        r"too deep", r"excessive depth", r"past parallel",
    ],
    "90 degrees": [
        r"parallel", r"good depth", r"adequate depth", r"at parallel",
        r"thighs.*parallel",
    ],
    "back not straight": [
        r"back.*round", r"rounding", r"lean", r"chest up", r"straight.*back",
        r"spine", r"lumbar", r"thoracic", r"forward lean", r"torso",
        r"butt wink", r"hyperextension", r"back.*not.*straight",
    ],
    "knees over toes": [
        r"knees.*toes", r"knees.*past", r"knees.*forward", r"shin angle",
        r"knees.*tracking", r"knee.*over", r"knee.*cave", r"knee.*valgus",
    ],
    "narrow": [
        r"narrow", r"closer", r"wider stance", r"feet.*apart",
        r"feet.*too close", r"stance.*narrow",
    ],
    "wide": [
        r"wide", r"feet.*too far", r"narrower", r"stance.*wide",
        r"too wide",
    ],
    "plie": [
        r"plie", r"sumo", r"turned out", r"ballet", r"toes.*out",
    ],
    "hold": [
        r"pause", r"hold", r"bottom.*position", r"isometric",
        r"paused squat",
    ],
    "insufficient": [
        r"incomplete", r"partial", r"insufficient", r"barely",
        r"minimal", r"not.*complete", r"half.*rep",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_dir", required=True, help="Dir with predictions.jsonl")
    p.add_argument("--data_path", required=True, help="Test data with ground truth")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def label_mentioned_in_coaching(label: str, coaching_text: str) -> bool:
    """Check if a label's issue is mentioned in coaching text."""
    if not coaching_text:
        return False
    patterns = LABEL_KEYWORDS.get(label, [])
    text_lower = coaching_text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def count_coaching_points(coaching_text: str) -> int:
    """Count distinct coaching points (numbered items, bullet points, or paragraphs)."""
    if not coaching_text:
        return 0
    # Count numbered items or bullet points
    items = re.findall(r"(?:^|\n)\s*(?:\d+[.):]|\-|\*|•)\s*\S", coaching_text)
    if items:
        return len(items)
    # Fallback: count non-empty paragraphs
    paragraphs = [p.strip() for p in coaching_text.split("\n\n") if p.strip()]
    return max(len(paragraphs), 1) if coaching_text.strip() else 0


def score_specificity(coaching_text: str) -> float:
    """Score coaching specificity 0-2.

    0 = generic ("work on your form")
    1 = specific issue identified ("your depth is shallow")
    2 = specific issue + actionable cue ("sit back more to reach parallel")
    """
    if not coaching_text or len(coaching_text) < 10:
        return 0.0

    text_lower = coaching_text.lower()

    # Check for actionable cues
    action_patterns = [
        r"try\s+\w+", r"focus\s+on", r"cue:", r"aim\s+to", r"work\s+on",
        r"squeeze", r"engage", r"push", r"pull", r"sit\s+back",
        r"keep\s+your", r"think\s+about", r"imagine", r"pretend",
        r"next\s+rep", r"practice", r"drill",
    ]
    has_action = any(re.search(p, text_lower) for p in action_patterns)

    # Check for specific issue identification
    has_specific = any(
        any(re.search(p, text_lower) for p in patterns)
        for patterns in LABEL_KEYWORDS.values()
    )

    if has_specific and has_action:
        return 2.0
    elif has_specific:
        return 1.0
    else:
        return 0.0


def check_contradiction(pred_labels: set, coaching_text: str) -> bool:
    """Check if coaching contradicts the classification."""
    if not coaching_text:
        return False
    text_lower = coaching_text.lower()

    # If classified as "90 degrees" but coaching says "shallow"
    if "90 degrees" in pred_labels:
        if re.search(r"shallow|not deep enough|more depth needed", text_lower):
            return True

    # If classified as "shoulder-width" but coaching says "too narrow"
    if "shoulder-width" in pred_labels:
        if re.search(r"too narrow|feet.*closer|narrow stance", text_lower):
            return True

    return False


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    preds = []
    pred_path = Path(args.predictions_dir) / "predictions.jsonl"
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    # Load ground truth
    with open(args.data_path) as f:
        gt_data = json.load(f)
    gt_map = {}
    for sample in gt_data:
        sid = sample.get("id", "")
        meta = sample.get("metadata", {})
        structured = meta.get("structured_labels", {})
        gt_labels = set()
        if structured:
            if structured.get("visible") is False:
                gt_labels = {"not visible"}
            else:
                if structured.get("stance"):
                    gt_labels.add(structured["stance"])
                if structured.get("depth"):
                    gt_labels.add(structured["depth"])
                for issue in structured.get("form_issues", []):
                    gt_labels.add(issue)
                if structured.get("variant"):
                    gt_labels.add(structured["variant"])
        gt_map[sid] = gt_labels

    # Evaluate coaching
    issue_labels = ["shallow", "over 90 degrees", "back not straight", "knees over toes",
                    "narrow", "wide", "plie", "hold", "insufficient"]

    per_issue = {l: {"total_in_gt": 0, "covered_in_coaching": 0, "times_hallucinated": 0}
                 for l in issue_labels}

    total_issue_recall = []
    total_issue_precision = []
    total_hallucination = []
    total_specificity = []
    total_contradiction = []
    total_coaching_points = []
    samples_with_no_coaching = 0

    for pred in preds:
        if "error" in pred and "pred_labels" not in pred:
            continue

        sid = pred["id"]
        coaching = pred.get("coaching", "")
        pred_labels = set(pred.get("pred_labels", pred.get("pred_labels_pass2", [])))
        gt_labels = gt_map.get(sid, set())

        if not coaching:
            samples_with_no_coaching += 1

        # Issue recall: fraction of GT issues mentioned in coaching
        gt_issues = gt_labels & set(issue_labels)
        if gt_issues:
            covered = sum(1 for l in gt_issues if label_mentioned_in_coaching(l, coaching))
            total_issue_recall.append(covered / len(gt_issues))
        else:
            total_issue_recall.append(1.0)  # No issues to recall = perfect

        # Per-label tracking
        for label in issue_labels:
            if label in gt_labels:
                per_issue[label]["total_in_gt"] += 1
                if label_mentioned_in_coaching(label, coaching):
                    per_issue[label]["covered_in_coaching"] += 1
            else:
                if label_mentioned_in_coaching(label, coaching):
                    per_issue[label]["times_hallucinated"] += 1

        # Issue precision: fraction of coaching points that match real issues
        coaching_mentions = set()
        for label in issue_labels:
            if label_mentioned_in_coaching(label, coaching):
                coaching_mentions.add(label)
        if coaching_mentions:
            correct_mentions = len(coaching_mentions & gt_labels)
            total_issue_precision.append(correct_mentions / len(coaching_mentions))
        else:
            total_issue_precision.append(1.0)  # No mentions = no wrong mentions

        # Hallucination rate
        hallucinated = coaching_mentions - gt_labels
        total_hallucination.append(len(hallucinated) / max(len(coaching_mentions), 1))

        # Specificity
        total_specificity.append(score_specificity(coaching))

        # Contradiction
        total_contradiction.append(1.0 if check_contradiction(pred_labels, coaching) else 0.0)

        # Coaching points
        total_coaching_points.append(count_coaching_points(coaching))

    n = len(total_issue_recall) or 1

    def _mean_se(values):
        """Compute mean and standard error (with Bessel's correction)."""
        if len(values) < 2:
            return {"mean": sum(values) / max(len(values), 1), "se": 0.0}
        m = sum(values) / len(values)
        var = sum((x - m) ** 2 for x in values) / (len(values) - 1)  # Bessel's correction
        se = math.sqrt(var) / math.sqrt(len(values))
        return {"mean": m, "se": se}

    results = {
        "n_samples": len(preds),
        "issue_recall": _mean_se(total_issue_recall),
        "issue_precision": _mean_se(total_issue_precision),
        "hallucination_rate": _mean_se(total_hallucination),
        "specificity": _mean_se(total_specificity),
        "contradiction_rate": _mean_se(total_contradiction),
        "avg_coaching_points": sum(total_coaching_points) / n,
        "samples_with_no_coaching": samples_with_no_coaching,
        "per_issue": per_issue,
    }

    with open(out_dir / "coaching_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"COACHING EVALUATION ({len(preds)} samples)")
    print(f"{'='*60}")
    print(f"Issue Recall:        {results['issue_recall']['mean']:.3f}")
    print(f"Issue Precision:     {results['issue_precision']['mean']:.3f}")
    print(f"Hallucination Rate:  {results['hallucination_rate']['mean']:.3f}")
    print(f"Specificity (0-2):   {results['specificity']['mean']:.3f}")
    print(f"Contradiction Rate:  {results['contradiction_rate']['mean']:.3f}")
    print(f"Avg Coaching Points: {results['avg_coaching_points']:.1f}")
    print(f"No Coaching:         {samples_with_no_coaching}/{len(preds)}")

    print(f"\nPer-issue breakdown:")
    for label, info in per_issue.items():
        if info["total_in_gt"] > 0:
            recall = info["covered_in_coaching"] / info["total_in_gt"]
            print(f"  {label}: recall={recall:.0%} ({info['covered_in_coaching']}/{info['total_in_gt']}), "
                  f"hallucinated={info['times_hallucinated']}")

    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
