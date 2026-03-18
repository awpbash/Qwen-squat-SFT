"""
v2: Always-tool two-pass inference (VLM → Pose → VLM refinement + coaching).

Changes from v1:
  - Generates coaching text in the refinement pass
  - Works on both QEVD and Reddit data
  - Saves both pass1 and pass2 predictions for analysis

Usage:
  python inference_with_pose.py --model_path <path> --data_path new_qevd_mrq_test.json \\
      --output_dir eval_results/E4_always_tool/controlled/
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
    ALL_LABELS, GEOMETRIC_LABELS, HOLISTIC_LABELS,
    load_model, extract_structured_from_text, structured_to_label_set,
    get_true_labels, compute_metrics, bootstrap_ci,
)

# Refinement prompt with coaching
REFINEMENT_PROMPT = """You previously assessed this squat video and gave:
{initial_json}

Now consider these objective pose measurements:
- Minimum knee angle: {knee_angle:.1f}° (depth indicator: >110°=shallow, 85-110°=~90°, <85°=deep)
- Maximum back deviation from vertical: {back_dev:.1f}° (>20° = back not straight)
- Stance width ratio (ankles/hips): {stance_ratio:.2f}x (<0.8=narrow, 0.8-1.5=shoulder, >1.5=wide, >2.0=plie)
- Knee past toes (normalized): {kot:.3f} (>0.08 = knees over toes)
- Hold duration: {hold:.1f}s (>1.0s = hold variant)
- Pose visibility: {visibility:.2f} (<0.3 = not visible)

Instructions:
1. Compare your visual assessment with the measurements. Where they conflict, explain your reasoning.
2. Output your REVISED classification JSON.
3. For each detected form issue, provide specific coaching:
   - What the issue is
   - Why it matters (injury risk or performance)
   - One concrete cue to fix it on the next rep

Output format:
CLASSIFICATION:
```json
{{"stance": "...", "depth": "...", "form_issues": [...], "variant": "..."|null, "visible": true|false}}
```

COACHING:
[Your specific coaching advice here]
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--video_root", default=".")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", default="eval_results/E4_always_tool")
    p.add_argument("--domain", choices=["controlled", "wild"], default="controlled")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--pose_workers", type=int, default=4)
    p.add_argument("--video_min_pixels", type=int, default=None)
    p.add_argument("--video_max_pixels", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def vlm_predict(model, processor, video_path, prompt, fps, device,
                video_min_pixels=None, video_max_pixels=None):
    """Single VLM inference pass."""
    video_info = {"type": "video", "video": video_path, "fps": fps}
    if video_min_pixels is not None:
        video_info["min_pixels"] = video_min_pixels
    if video_max_pixels is not None:
        video_info["max_pixels"] = video_max_pixels

    messages = [{
        "role": "user",
        "content": [
            video_info,
            {"type": "text", "text": prompt.replace("<video>", "").replace("<video>\n", "").strip()},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    gen_ids = model.generate(**inputs, max_new_tokens=768, do_sample=False, temperature=0.0)
    gen = gen_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def vlm_refine(model, processor, video_path, refinement_text, fps, device,
               video_min_pixels=None, video_max_pixels=None):
    """Second VLM pass with pose measurements injected."""
    video_info = {"type": "video", "video": video_path, "fps": fps}
    if video_min_pixels is not None:
        video_info["min_pixels"] = video_min_pixels
    if video_max_pixels is not None:
        video_info["max_pixels"] = video_max_pixels

    messages = [{
        "role": "user",
        "content": [
            video_info,
            {"type": "text", "text": refinement_text},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    gen_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, temperature=0.0)
    gen = gen_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def extract_coaching(text: str) -> str:
    """Extract coaching section from refinement output."""
    # Look for COACHING: header
    patterns = [r"COACHING:\s*\n(.*)", r"coaching:\s*\n(.*)", r"\*\*Coaching.*?\*\*\s*\n(.*)"]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: everything after the JSON block
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

    print(f"Always-tool two-pass eval: {len(data)} samples [{args.domain}]")

    model, processor = load_model(args.model_path, args.base_model, device)
    prompt = data[0]["conversations"][0]["value"]

    # Pre-compute all pose analyses in parallel
    print("Running pose analysis in parallel...")
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

    # Two-pass inference
    y_true_list = []
    y_pred_list_p1 = []
    y_pred_list_p2 = []
    predictions = []
    changes = {"improved": 0, "degraded": 0, "unchanged": 0, "changed_total": 0}

    for sample in tqdm(data, desc="VLM two-pass"):
        sid = sample.get("id", "")
        video_rel = sample.get("video", "")
        video_abs = str((Path(args.video_root) / video_rel).resolve())
        true_labels = get_true_labels(sample)

        if not Path(video_abs).exists():
            predictions.append({"id": sid, "error": "video_missing"})
            continue

        # Pass 1: VLM classification
        try:
            raw_p1 = vlm_predict(model, processor, video_abs, prompt, args.fps, device,
                                 args.video_min_pixels, args.video_max_pixels)
            parsed_p1 = extract_structured_from_text(raw_p1)
            labels_p1 = structured_to_label_set(parsed_p1)
        except Exception as e:
            raw_p1 = f"ERROR: {e}"
            parsed_p1 = None
            labels_p1 = set()

        # Get pose measurements
        pose = pose_results.get(sid, {"error": "not_computed"})

        # Pass 2: Refinement with pose + coaching
        coaching = ""
        if not pose.get("error") and parsed_p1 is not None:
            initial_json = json.dumps(parsed_p1) if parsed_p1 else "{}"
            refinement_text = REFINEMENT_PROMPT.format(
                initial_json=initial_json,
                knee_angle=pose.get("knee_angle_min", 0),
                back_dev=pose.get("back_deviation_max_deg", 0),
                stance_ratio=pose.get("stance_ratio", 0),
                kot=pose.get("knee_over_toe_normalized", 0),
                hold=pose.get("hold_duration_sec", 0),
                visibility=pose.get("avg_landmark_visibility", 0),
            )

            try:
                raw_p2 = vlm_refine(model, processor, video_abs, refinement_text, args.fps, device,
                                    args.video_min_pixels, args.video_max_pixels)
                parsed_p2 = extract_structured_from_text(raw_p2)
                labels_p2 = structured_to_label_set(parsed_p2)
                coaching = extract_coaching(raw_p2)
            except Exception as e:
                raw_p2 = f"ERROR: {e}"
                parsed_p2 = parsed_p1
                labels_p2 = labels_p1
        else:
            raw_p2 = raw_p1
            parsed_p2 = parsed_p1
            labels_p2 = labels_p1

        # Track changes (exact match: did the full label set become correct?)
        if labels_p1 != labels_p2:
            changes["changed_total"] += 1
            p1_exact = labels_p1 == true_labels
            p2_exact = labels_p2 == true_labels
            if p2_exact and not p1_exact:
                changes["improved"] += 1
            elif p1_exact and not p2_exact:
                changes["degraded"] += 1
            else:
                # Both wrong or both right (with different predictions) — use Jaccard
                def _jaccard(pred, gt):
                    union = pred | gt
                    return len(pred & gt) / len(union) if union else 1.0
                j1 = _jaccard(labels_p1, true_labels)
                j2 = _jaccard(labels_p2, true_labels)
                if j2 > j1:
                    changes["improved"] += 1
                elif j2 < j1:
                    changes["degraded"] += 1
                else:
                    changes["unchanged"] += 1

        y_true_list.append(true_labels)
        y_pred_list_p1.append(labels_p1)
        y_pred_list_p2.append(labels_p2)

        predictions.append({
            "id": sid,
            "video": video_rel,
            "true_labels": sorted(true_labels),
            "pred_labels_pass1": sorted(labels_p1),
            "pred_labels_pass2": sorted(labels_p2),
            "prediction_raw_pass1": raw_p1,
            "prediction_raw_pass2": raw_p2,
            "coaching": coaching,
            "pose_measurements": {k: v for k, v in pose.items()
                                  if k not in ("suggested_labels", "per_frame", "error")}
                if not pose.get("error") else None,
            "domain": args.domain,
        })

    # Compute metrics for both passes
    metrics_p1 = compute_metrics(y_true_list, y_pred_list_p1)
    metrics_p2 = compute_metrics(y_true_list, y_pred_list_p2)

    print("Computing bootstrap CI for pass 2...")
    ci = bootstrap_ci(y_true_list, y_pred_list_p2)

    result = {
        "pass1_metrics": metrics_p1,
        "pass2_metrics": metrics_p2,
        "changes": changes,
        "bootstrap_ci": ci,
        "domain": args.domain,
        "total_samples": len(y_true_list),
    }

    # For compatibility, top-level metrics = pass2
    result.update(metrics_p2)

    # Save
    with open(out_dir / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print
    print(f"\n{'='*70}")
    print(f"ALWAYS-TOOL RESULTS — {args.domain.upper()} ({len(y_true_list)} samples)")
    print(f"{'='*70}")
    print(f"Pass 1 Macro F1: {metrics_p1['macro']['f1']:.4f}")
    print(f"Pass 2 Macro F1: {metrics_p2['macro']['f1']:.4f}  [{ci['macro_f1']['ci_low']:.3f}, {ci['macro_f1']['ci_high']:.3f}]")
    print(f"Changes: {changes['changed_total']} total — "
          f"{changes['improved']} improved, {changes['degraded']} degraded")
    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
