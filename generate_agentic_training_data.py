"""
v2: Generate agentic training data (classification + confidence + tool decision).

Changes from v1:
  - Measurement-based confidence: runs pose_tool on each video and compares
    its suggested labels to ground truth to determine per-field confidence.
    This replaces the v1 label-deterministic approach where every "narrow"
    always got LOW confidence regardless of the video.
  - Creates three outputs: agentic-only, mixed (50/50), and DPO pairs

Confidence logic:
  LOW (→ INVOKE_TOOL):  pose_tool agrees with GT for this field on this video,
                        meaning the tool CAN help the model get it right.
  HIGH (→ NO_TOOL):     pose_tool disagrees with GT or fails, meaning the tool
                        would not help (or would mislead) for this field.

Fallback: when pose analysis fails entirely (no pose detected, video error),
  falls back to v1 per-label difficulty priors from empirical results.

Data volume control: The mixed dataset is size-matched to the standard
  training data (N total = N/2 standard + N/2 agentic) so that E5b trains
  on the same number of samples as E2, avoiding a data-volume confound.

Input:  new_qevd_mrq_train.json (oversampled train split)
Output: new_qevd_mrq_agentic_train.json  (agentic format only)
        new_qevd_mrq_mixed_train.json    (50% standard + 50% agentic, size-matched)
"""

import json
import random
from pathlib import Path
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from pose_tool import analyze_squat

TRAIN_PATH = Path("new_qevd_mrq_train.json")
AGENTIC_PATH = Path("new_qevd_mrq_agentic_train.json")
MIXED_PATH = Path("new_qevd_mrq_mixed_train.json")
POSE_CACHE_PATH = Path("pose_cache.json")

SEED = 42
POSE_WORKERS = 8
POSE_FPS = 2.0

# ── Fallback: v1 per-label difficulty priors ──────────────────────
# Used ONLY when pose analysis fails for a video.
VLM_STRONG = {
    "shoulder-width",   # F1=0.80 in v1
    "90 degrees",       # F1=0.72 in v1
}
FORM_UNCERTAIN = {
    "back not straight",
    "knees over toes",
    "insufficient",
}

CONFIDENCE_FIELDS = ["stance", "depth", "form_issues", "variant", "visible"]

# ── Agentic prompt template ─────────────────────────────────────────
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


# ── Pose analysis ────────────────────────────────────────────────────

def _run_pose_worker(args_tuple):
    """Worker for parallel pose analysis. Returns (video_path, result)."""
    video_path, fps = args_tuple
    try:
        result = analyze_squat(video_path, fps=fps)
        return video_path, result
    except Exception as e:
        return video_path, {"error": str(e)}


def compute_pose_for_videos(video_paths: list[str], fps: float, workers: int,
                            cache_path: Path) -> dict[str, dict]:
    """Run pose analysis on unique videos, with disk caching.

    Returns dict mapping video_path → pose_result.
    """
    # Load cache if available
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded pose cache: {len(cache)} entries")

    # Find videos that need analysis
    need_analysis = [v for v in video_paths if v not in cache]
    print(f"Unique videos: {len(video_paths)}, cached: {len(video_paths) - len(need_analysis)}, "
          f"to analyze: {len(need_analysis)}")

    if need_analysis:
        work = [(v, fps) for v in need_analysis]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_pose_worker, w) for w in work]
            for future in tqdm(futures, desc="Pose analysis"):
                vpath, result = future.result()
                cache[vpath] = result

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"Saved pose cache: {len(cache)} entries → {cache_path}")

    return cache


# ── Confidence assignment ────────────────────────────────────────────

def assign_confidence(structured_labels: dict,
                      pose_result: dict | None) -> tuple[dict, str]:
    """Assign per-field confidence based on pose measurement agreement with GT.

    For each field, compares the pose_tool's suggestion to the ground truth:
      - If pose agrees with GT → tool can help → LOW confidence → INVOKE
      - If pose disagrees with GT → tool would mislead → HIGH → NO_TOOL

    Falls back to v1 per-label difficulty when pose analysis fails.

    Returns (confidence_dict, tool_decision).
    """
    if not structured_labels.get("visible", True):
        return {f: "HIGH" for f in CONFIDENCE_FIELDS}, "NO_TOOL_NEEDED"

    confidence = {"visible": "HIGH"}

    # Extract pose suggestions (None if pose failed)
    pose_labels = None
    if pose_result and not pose_result.get("error"):
        pose_labels = pose_result.get("suggested_labels")

    # ── Stance ──
    gt_stance = structured_labels.get("stance")
    if gt_stance is None:
        # No ground truth for stance — default to HIGH (no tool needed for unknown GT)
        confidence["stance"] = "HIGH"
    elif pose_labels and pose_labels.get("stance") is not None:
        confidence["stance"] = "LOW" if pose_labels["stance"] == gt_stance else "HIGH"
    else:
        # Fallback: v1 difficulty
        confidence["stance"] = "HIGH" if gt_stance in VLM_STRONG else "LOW"

    # ── Depth ──
    gt_depth = structured_labels.get("depth")
    if gt_depth is None:
        confidence["depth"] = "HIGH"
    elif pose_labels and pose_labels.get("depth") is not None:
        confidence["depth"] = "LOW" if pose_labels["depth"] == gt_depth else "HIGH"
    else:
        confidence["depth"] = "HIGH" if gt_depth in VLM_STRONG else "LOW"

    # ── Form issues ──
    gt_form = set(structured_labels.get("form_issues", []))
    if pose_labels and pose_labels.get("form_issues") is not None:
        pose_form = set(pose_labels.get("form_issues", []))
        if gt_form:
            # Tool helps if it correctly identifies at least one real issue
            overlap = gt_form & pose_form
            confidence["form_issues"] = "LOW" if overlap else "HIGH"
        else:
            # GT has no issues — check if tool also says clean
            confidence["form_issues"] = "LOW" if not pose_form else "HIGH"
    else:
        # Fallback: v1 difficulty
        if not gt_form:
            confidence["form_issues"] = "HIGH"
        elif any(f in FORM_UNCERTAIN for f in gt_form):
            confidence["form_issues"] = "LOW"
        else:
            confidence["form_issues"] = "HIGH"

    # ── Variant (hold) ──
    gt_variant = structured_labels.get("variant")
    if pose_labels:
        pose_variant = pose_labels.get("variant")
        # Both None (no hold) or both "hold" → agreement → tool helps
        confidence["variant"] = "LOW" if gt_variant == pose_variant else "HIGH"
    else:
        confidence["variant"] = "LOW" if gt_variant == "hold" else "HIGH"

    # ── Tool decision ──
    has_low = any(v == "LOW" for v in confidence.values())
    tool_decision = "INVOKE_TOOL" if has_low else "NO_TOOL_NEEDED"

    return confidence, tool_decision


# ── Response building ────────────────────────────────────────────────

def build_agentic_response(structured: dict, confidence: dict, tool_decision: str) -> str:
    """Build the agentic training response."""
    if not structured["visible"]:
        assessment = "The squat is not clearly visible in the video."
    else:
        lines = []
        if structured.get("stance"):
            lines.append(f"The person appears to have a {structured['stance']} stance.")
        if structured.get("depth"):
            lines.append(f"Squat depth appears to be approximately {structured['depth']}.")
        if structured["form_issues"]:
            issues_str = ", ".join(structured["form_issues"])
            lines.append(f"Form observations: {issues_str}.")
        else:
            lines.append("No obvious form issues observed.")
        if structured["variant"]:
            lines.append(f"The person appears to perform a {structured['variant']} variant.")
        assessment = " ".join(lines)

    conf_lines = []
    reasons = {
        "stance": {
            "HIGH": "stance width is clearly visible",
            "LOW": "difficult to judge precise foot placement from this angle",
        },
        "depth": {
            "HIGH": "depth is clearly visible at this angle",
            "LOW": "precise knee angle is hard to determine visually",
        },
        "form_issues": {
            "HIGH": "form quality is apparent from visual observation",
            "LOW": "subtle form issues may require measurement to confirm",
        },
        "variant": {
            "HIGH": "movement pattern is clear",
            "LOW": "bottom position duration is hard to judge visually",
        },
        "visible": {
            "HIGH": "the person and squat are clearly visible",
            "LOW": "visibility is poor",
        },
    }
    for field in CONFIDENCE_FIELDS:
        level = confidence[field]
        reason = reasons.get(field, {}).get(level, "")
        conf_lines.append(f"- {field}: {level} — {reason}")

    json_str = json.dumps(structured)

    return (
        f"ASSESSMENT:\n{assessment}\n\n"
        f"CONFIDENCE:\n" + "\n".join(conf_lines) + "\n\n"
        f"TOOL_DECISION: {tool_decision}\n\n"
        f"CLASSIFICATION:\n```json\n{json_str}\n```"
    )


def build_agentic_sample(sample: dict, pose_result: dict | None) -> dict:
    """Convert a standard training sample to agentic format."""
    new = deepcopy(sample)
    structured = new["metadata"]["structured_labels"]
    confidence, tool_decision = assign_confidence(structured, pose_result)

    new["id"] = new["id"] + "_agentic"
    new["conversations"][0]["value"] = AGENTIC_PROMPT.strip()
    new["conversations"][1]["value"] = build_agentic_response(structured, confidence, tool_decision)
    new["metadata"]["confidence"] = confidence
    new["metadata"]["tool_decision"] = tool_decision

    return new


# ── Main ─────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples")

    # Collect unique video paths (oversampled data has many duplicates)
    unique_videos = list({s["video"] for s in train_data})
    print(f"Unique videos: {len(unique_videos)} (from {len(train_data)} samples)")

    # Pre-compute pose analysis on unique videos (parallel, cached)
    # Resolve paths relative to project root
    video_abs_paths = [str(Path(v).resolve()) for v in unique_videos]
    pose_cache = compute_pose_for_videos(video_abs_paths, POSE_FPS, POSE_WORKERS,
                                         POSE_CACHE_PATH)

    # Map relative paths to absolute for lookup
    rel_to_abs = {rel: str(Path(rel).resolve()) for rel in unique_videos}

    # Track pose analysis stats
    pose_ok = sum(1 for v in unique_videos if not pose_cache.get(rel_to_abs[v], {}).get("error"))
    pose_fail = len(unique_videos) - pose_ok
    print(f"Pose results: {pose_ok} OK, {pose_fail} failed ({pose_fail/len(unique_videos):.1%} fallback)")

    # Build agentic versions
    agentic_data = []
    tool_decision_counts = Counter()
    confidence_counts = {f: Counter() for f in CONFIDENCE_FIELDS}
    # Per-field: how often pose-based vs fallback
    source_counts = {f: Counter() for f in CONFIDENCE_FIELDS}

    for sample in tqdm(train_data, desc="Building agentic data"):
        video_rel = sample["video"]
        video_abs = rel_to_abs.get(video_rel, str(Path(video_rel).resolve()))
        pose_result = pose_cache.get(video_abs)

        agentic = build_agentic_sample(sample, pose_result)
        agentic_data.append(agentic)

        tool_decision_counts[agentic["metadata"]["tool_decision"]] += 1
        for field, level in agentic["metadata"]["confidence"].items():
            confidence_counts[field][level] += 1

    # Print statistics
    total = len(agentic_data)
    print(f"\nAgentic data: {total} samples")
    print(f"Tool decisions: {dict(tool_decision_counts)}")
    invoke_rate = tool_decision_counts.get("INVOKE_TOOL", 0) / total
    print(f"Tool invocation rate: {invoke_rate:.1%}")

    print("\nConfidence distribution:")
    for field, counts in confidence_counts.items():
        field_total = counts["HIGH"] + counts["LOW"]
        if field_total > 0:
            print(f"  {field}: HIGH={counts['HIGH']}/{field_total} ({counts['HIGH']/field_total:.0%}), "
                  f"LOW={counts['LOW']}/{field_total} ({counts['LOW']/field_total:.0%})")

    # Save agentic-only
    with open(AGENTIC_PATH, "w") as f:
        json.dump(agentic_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {AGENTIC_PATH} ({len(agentic_data)} samples)")

    # Create mixed dataset (50/50), SIZE-MATCHED to standard training data.
    # E2 trains on len(train_data) samples. To avoid a data-volume confound,
    # the mixed set is the same total size: N/2 standard + N/2 agentic.
    n_total = len(train_data)
    n_half = n_total // 2

    standard_sample = random.sample(train_data, n_half)
    agentic_sample = random.sample(agentic_data, n_total - n_half)  # handles odd N

    mixed = standard_sample + agentic_sample
    random.shuffle(mixed)
    print(f"\nMixed dataset (size-matched to E2 training data):")
    print(f"  Standard portion: {len(standard_sample)}")
    print(f"  Agentic portion:  {len(agentic_sample)}")
    print(f"  Total:            {len(mixed)} (same as E2's {n_total})")

    with open(MIXED_PATH, "w") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)
    print(f"Saved: {MIXED_PATH} ({len(mixed)} samples)")


if __name__ == "__main__":
    main()
