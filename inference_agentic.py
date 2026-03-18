"""
v2: Agentic inference — VLM decides when to invoke the pose tool.

Changes from v1:
  - Generates coaching in both tool/no-tool paths
  - Tracks per-field confidence distribution and calibration
  - Works on both QEVD and Reddit data
  - Cleaner parsing of confidence/tool-decision from output

Usage:
  python inference_agentic.py --model_path <path> --data_path new_qevd_mrq_test.json \\
      --output_dir eval_results/E5_agentic/controlled/
"""

import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

from pose_tool import analyze_squat
from eval_unified import (
    ALL_LABELS, STANCE_LABELS, DEPTH_LABELS, FORM_LABELS,
    GEOMETRIC_LABELS, HOLISTIC_LABELS,
    load_model, extract_structured_from_text, structured_to_label_set,
    get_true_labels, compute_metrics, bootstrap_ci,
)

# ── Agentic prompt (Pass 1) ────────────────────────────────────────
AGENTIC_PROMPT = """<video>
You are a squat form coach with access to a measurement tool.

Available tool:
  get_pose_measurements(video) → returns knee angle, back deviation,
  stance ratio, knee-over-toe displacement, hold duration.

Instructions:
1. Watch the video carefully. Describe what you observe about stance, depth, and form.
2. For EACH field below, state your confidence level:
   - stance: HIGH or LOW
   - depth: HIGH or LOW
   - form_issues: HIGH or LOW
   - variant: HIGH or LOW
   - visible: HIGH or LOW
3. Make your tool decision:
   - If ANY field is LOW confidence → output TOOL_DECISION: INVOKE_TOOL
   - If ALL fields are HIGH confidence → output TOOL_DECISION: NO_TOOL_NEEDED
4. Output your classification JSON.

Output format:
ASSESSMENT:
[Your observations]

CONFIDENCE:
- stance: [HIGH/LOW] — [brief reason]
- depth: [HIGH/LOW] — [brief reason]
- form_issues: [HIGH/LOW] — [brief reason]
- variant: [HIGH/LOW] — [brief reason]
- visible: [HIGH/LOW] — [brief reason]

TOOL_DECISION: [INVOKE_TOOL / NO_TOOL_NEEDED]

CLASSIFICATION:
```json
{"stance": "...", "depth": "...", "form_issues": [...], "variant": "..."|null, "visible": true|false}
```
"""

# ── Refinement prompt (Pass 2, if tool invoked) ────────────────────
AGENTIC_REFINEMENT = """You invoked get_pose_measurements. Here are the results:

- Minimum knee angle: {knee_angle:.1f}° (>110°=shallow, 85-110°≈90°, <85°=deep)
- Maximum back deviation: {back_dev:.1f}° (>20° = back not straight)
- Stance ratio (ankles/hips): {stance_ratio:.2f}x (<0.8=narrow, 0.8-1.5=shoulder, >1.5=wide, >2.0=plie)
- Knee past toes: {kot:.3f} (>0.08 = knees over toes)
- Hold duration: {hold:.1f}s (>1.0s = hold variant)

Your initial classification was:
{initial_json}

Instructions:
1. Compare measurements with your visual assessment. Where they conflict, explain your reasoning.
2. Output your REVISED classification JSON.
3. Provide specific, actionable coaching for each detected issue.

REVISED CLASSIFICATION:
```json
{{"stance": "...", "depth": "...", "form_issues": [...], "variant": "..."|null, "visible": true|false}}
```

COACHING:
[Specific coaching for each issue]
"""

# ── Coaching-only prompt (if no tool needed) ───────────────────────
COACHING_ONLY = """Based on your classification:
{initial_json}

Provide specific, actionable coaching for each detected form issue.
For each issue:
- What the problem is
- Why it matters (injury risk or performance)
- One concrete cue to try on the next rep

If no issues were detected, provide brief positive reinforcement.

COACHING:
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--video_root", default=".")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", default="eval_results/E5_agentic")
    p.add_argument("--domain", choices=["controlled", "wild"], default="controlled")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--pose_workers", type=int, default=4)
    p.add_argument("--video_min_pixels", type=int, default=None)
    p.add_argument("--video_max_pixels", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def vlm_generate(model, processor, video_path, text_prompt, fps, device,
                 max_tokens=1024, video_min_pixels=None, video_max_pixels=None):
    video_info = {"type": "video", "video": video_path, "fps": fps}
    if video_min_pixels is not None:
        video_info["min_pixels"] = video_min_pixels
    if video_max_pixels is not None:
        video_info["max_pixels"] = video_max_pixels

    messages = [{
        "role": "user",
        "content": [
            video_info,
            {"type": "text", "text": text_prompt.replace("<video>", "").replace("<video>\n", "").strip()},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    gen_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.0)
    gen = gen_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def vlm_text_only(model, processor, text_prompt, device, max_tokens=512):
    """Text-only generation (no video) for coaching follow-up."""
    messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    gen_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.0)
    gen = gen_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_confidence(text: str) -> dict:
    """Extract per-field confidence from agentic output."""
    fields = ["stance", "depth", "form_issues", "variant", "visible"]
    confidence = {}
    for field in fields:
        pattern = rf"{field}:\s*(HIGH|LOW)"
        m = re.search(pattern, text, re.IGNORECASE)
        confidence[field] = m.group(1).upper() if m else "HIGH"  # Default HIGH if not found
    return confidence


def parse_tool_decision(text: str) -> str:
    """Extract TOOL_DECISION from agentic output."""
    m = re.search(r"TOOL_DECISION:\s*(INVOKE_TOOL|NO_TOOL_NEEDED)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for keywords
    if "invoke" in text.lower() and "tool" in text.lower():
        return "INVOKE_TOOL"
    return "NO_TOOL_NEEDED"


def extract_coaching(text: str) -> str:
    patterns = [r"COACHING:\s*\n?(.*)", r"coaching:\s*\n?(.*)", r"\*\*Coaching.*?\*\*\s*\n(.*)"]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    json_end = text.rfind("```")
    if json_end > 0 and json_end < len(text) - 10:
        return text[json_end + 3:].strip()
    return ""


def run_pose_worker(args_tuple):
    sid, video_abs, fps = args_tuple
    try:
        return sid, analyze_squat(video_abs, fps=fps)
    except Exception as e:
        return sid, {"error": str(e)}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Agentic eval: {len(data)} samples [{args.domain}]")
    model, processor = load_model(args.model_path, args.base_model, device)

    # Pre-compute pose for all videos (needed if tool is invoked)
    print("Pre-computing pose measurements...")
    pose_work = []
    for sample in data:
        sid = sample.get("id", "")
        video_rel = sample.get("video", "")
        video_abs = str((Path(args.video_root) / video_rel).resolve())
        if Path(video_abs).exists():
            pose_work.append((sid, video_abs, args.fps))

    pose_results = {}
    with ProcessPoolExecutor(max_workers=args.pose_workers) as pool:
        futures = [pool.submit(run_pose_worker, w) for w in pose_work]
        for future in tqdm(futures, desc="Pose"):
            sid, result = future.result()
            pose_results[sid] = result

    # Agentic inference
    y_true_list = []
    y_pred_list = []
    predictions = []
    confidence_dist = {f: {"HIGH": 0, "LOW": 0} for f in ["stance", "depth", "form_issues", "variant", "visible"]}
    tool_invoked_count = 0

    for sample in tqdm(data, desc="Agentic inference"):
        sid = sample.get("id", "")
        video_rel = sample.get("video", "")
        video_abs = str((Path(args.video_root) / video_rel).resolve())
        true_labels = get_true_labels(sample)

        if not Path(video_abs).exists():
            predictions.append({"id": sid, "error": "video_missing"})
            continue

        # Pass 1: Agentic assessment
        try:
            raw_p1 = vlm_generate(model, processor, video_abs, AGENTIC_PROMPT,
                                  args.fps, device, max_tokens=1024,
                                  video_min_pixels=args.video_min_pixels,
                                  video_max_pixels=args.video_max_pixels)
        except Exception as e:
            raw_p1 = f"ERROR: {e}"

        confidence = parse_confidence(raw_p1)
        tool_decision = parse_tool_decision(raw_p1)
        parsed_p1 = extract_structured_from_text(raw_p1)
        labels_p1 = structured_to_label_set(parsed_p1)

        # Track confidence distribution
        for field, level in confidence.items():
            confidence_dist[field][level] += 1

        # Conditional Pass 2
        coaching = ""
        final_labels = labels_p1
        final_raw = raw_p1
        raw_p2 = ""

        if tool_decision == "INVOKE_TOOL":
            tool_invoked_count += 1
            pose = pose_results.get(sid, {"error": "not_computed"})

            if not pose.get("error") and parsed_p1 is not None:
                refinement_text = AGENTIC_REFINEMENT.format(
                    initial_json=json.dumps(parsed_p1),
                    knee_angle=pose.get("knee_angle_min", 0),
                    back_dev=pose.get("back_deviation_max_deg", 0),
                    stance_ratio=pose.get("stance_ratio", 0),
                    kot=pose.get("knee_over_toe_normalized", 0),
                    hold=pose.get("hold_duration_sec", 0),
                )
                try:
                    raw_p2 = vlm_generate(model, processor, video_abs, refinement_text,
                                          args.fps, device, max_tokens=1024,
                                          video_min_pixels=args.video_min_pixels,
                                          video_max_pixels=args.video_max_pixels)
                    parsed_p2 = extract_structured_from_text(raw_p2)
                    if parsed_p2:
                        final_labels = structured_to_label_set(parsed_p2)
                    coaching = extract_coaching(raw_p2)
                    final_raw = raw_p2
                except Exception:
                    pass
        else:
            # No tool — generate coaching from classification only
            if parsed_p1 is not None:
                coaching_prompt = COACHING_ONLY.format(initial_json=json.dumps(parsed_p1))
                try:
                    coaching = vlm_text_only(model, processor, coaching_prompt, device)
                except Exception:
                    pass

        y_true_list.append(true_labels)
        y_pred_list.append(final_labels)

        predictions.append({
            "id": sid,
            "video": video_rel,
            "true_labels": sorted(true_labels),
            "pred_labels": sorted(final_labels),
            "confidence": confidence,
            "tool_decision": tool_decision,
            "coaching": coaching,
            "prediction_raw_pass1": raw_p1,
            "prediction_raw_pass2": raw_p2 if tool_decision == "INVOKE_TOOL" else "",
            "domain": args.domain,
        })

    # Compute metrics
    metrics = compute_metrics(y_true_list, y_pred_list)
    n = len(y_true_list) or 1
    metrics["tool_invocation_rate"] = tool_invoked_count / n
    metrics["domain"] = args.domain
    metrics["total_samples"] = n
    metrics["confidence_distribution"] = confidence_dist

    # Calibration analysis
    calibration = {}
    for group_name, group_labels, group_type in [
        ("stance", STANCE_LABELS, "geometric"),
        ("depth", DEPTH_LABELS, "geometric"),
        ("form", FORM_LABELS, "holistic"),
    ]:
        group_set = set(group_labels)
        tool_invoked_correct = 0
        tool_invoked_wrong = 0
        no_tool_correct = 0
        no_tool_wrong = 0

        for pred_rec, true_set, pred_set in zip(predictions, y_true_list, y_pred_list):
            if "error" in pred_rec:
                continue
            tool_used = pred_rec.get("tool_decision") == "INVOKE_TOOL"
            true_group = true_set & group_set
            pred_group = pred_set & group_set
            correct = (true_group == pred_group)

            if tool_used:
                if correct:
                    tool_invoked_correct += 1
                else:
                    tool_invoked_wrong += 1
            else:
                if correct:
                    no_tool_correct += 1
                else:
                    no_tool_wrong += 1

        total = tool_invoked_correct + tool_invoked_wrong + no_tool_correct + no_tool_wrong
        no_tool_total = no_tool_correct + no_tool_wrong
        calibration[group_name] = {
            "type": group_type,
            "tool_invocation_rate": (tool_invoked_correct + tool_invoked_wrong) / total if total else 0,
            "accuracy_when_not_invoked": no_tool_correct / no_tool_total if no_tool_total else 0,
            "missed_opportunity_rate": no_tool_wrong / no_tool_total if no_tool_total else 0,
            "counts": {
                "tool_invoked_correct": tool_invoked_correct,
                "tool_invoked_wrong": tool_invoked_wrong,
                "no_tool_correct": no_tool_correct,
                "no_tool_wrong": no_tool_wrong,
            },
        }

    metrics["calibration"] = calibration

    print("Computing bootstrap CI...")
    metrics["bootstrap_ci"] = bootstrap_ci(y_true_list, y_pred_list)

    # Save
    with open(out_dir / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    with open(out_dir / "calibration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "type", "tool_invocation_rate", "acc_when_not_invoked", "missed_opportunity_rate"])
        for g, c in calibration.items():
            writer.writerow([g, c["type"], f"{c['tool_invocation_rate']:.4f}",
                           f"{c['accuracy_when_not_invoked']:.4f}", f"{c['missed_opportunity_rate']:.4f}"])

    # Print
    ci = metrics["bootstrap_ci"]
    print(f"\n{'='*70}")
    print(f"AGENTIC RESULTS — {args.domain.upper()} ({n} samples)")
    print(f"{'='*70}")
    print(f"Tool invocation rate: {metrics['tool_invocation_rate']:.1%} ({tool_invoked_count}/{n})")
    print(f"Macro F1:        {metrics['macro']['f1']:.4f}  [{ci['macro_f1']['ci_low']:.3f}, {ci['macro_f1']['ci_high']:.3f}]")
    print(f"Geometric F1:    {metrics['geometric_avg_f1']:.4f}")
    print(f"Holistic F1:     {metrics['holistic_avg_f1']:.4f}")
    print(f"\nCalibration:")
    for g, c in calibration.items():
        print(f"  {g} ({c['type']}): invoked={c['tool_invocation_rate']:.1%}, "
              f"acc_no_tool={c['accuracy_when_not_invoked']:.1%}, "
              f"missed_opp={c['missed_opportunity_rate']:.1%}")
    print(f"\nConfidence distribution:")
    for field, dist in confidence_dist.items():
        total = dist["HIGH"] + dist["LOW"]
        print(f"  {field}: HIGH={dist['HIGH']}/{total} ({dist['HIGH']/total:.0%}), LOW={dist['LOW']}/{total}")
    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
