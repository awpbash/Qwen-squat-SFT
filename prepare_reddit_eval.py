"""
Prepare Reddit in-the-wild squat videos for evaluation.

Converts the Reddit biomechanics-label format into the same QEVD structured
format used by eval_unified.py, so all models can be evaluated on wild data
using the same pipeline.

Input:  archive/reddit_eval/export_pack/reddit_squat_eval_with_qevd_labels.jsonl
        archive/reddit_eval/export_pack/reps/  (70 video files)
Output: reddit_eval_test.json  (same schema as new_qevd_mrq_test.json)
        reddit_videos/         (symlinked video dir for consistent paths)
"""

import json
from pathlib import Path
from collections import Counter

INPUT_JSONL = Path("archive/reddit_eval/export_pack/reddit_squat_eval_with_qevd_labels.jsonl")
VIDEO_DIR = Path("archive/reddit_eval/export_pack/reps")
OUTPUT_PATH = Path("reddit_eval_test.json")
VIDEO_LINK_DIR = Path("reddit_videos")

# Map from QEVD flat labels to structured (shared)
from constants import STANCE_MAP, DEPTH_MAP, FORM_MAP

# Map Reddit biomechanics labels to QEVD equivalents
REDDIT_TO_QEVD = {
    "insufficient_depth": "squats - shallow",
    "back_rounding": "squats - back not straight",
    "forward_lean": "squats - back not straight",       # Closest mapping
    "knee_valgus": "squats - knees over toes",          # Closest mapping
    "heel_lift": None,                                   # No QEVD equivalent
    "butt_wink": "squats - back not straight",           # Related to back form
    "torso_collapse": "squats - back not straight",      # Related to back form
    "loss_of_bracing": None,                             # No QEVD equivalent
    "unstable_feet": None,                               # No QEVD equivalent
    "back_hyperextension": "squats - back not straight", # Related to back form
    "none": None,                                        # No issues
}

# Training prompt (identical to QEVD eval)
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


def qevd_labels_to_structured(qevd_labels: list[str]) -> dict:
    """Convert QEVD flat labels to structured format."""
    label_set = set(qevd_labels)
    if "squats - not visible" in label_set:
        return {"stance": None, "depth": None, "form_issues": None, "variant": None, "visible": False}

    stance = next((STANCE_MAP[l] for l in STANCE_MAP if l in label_set), None)
    depth = next((DEPTH_MAP[l] for l in DEPTH_MAP if l in label_set), None)
    form_issues = sorted([FORM_MAP[l] for l in FORM_MAP if l in label_set])
    variant = "hold" if "squats - hold" in label_set else None

    return {
        "stance": stance,
        "depth": depth,
        "form_issues": form_issues,
        "variant": variant,
        "visible": True,
    }


def convert_reddit_sample(reddit_sample: dict) -> dict:
    """Convert a Reddit sample to the QEVD evaluation format."""
    sid = reddit_sample["id"]
    video_name = reddit_sample.get("video_path", reddit_sample.get("targets", {}).get("video_path", ""))

    # Get QEVD-mapped labels (already mapped in the source file)
    qevd_labels = reddit_sample.get("qevd_labels_mapped", [])

    # If no pre-mapped labels, map from Reddit issues
    if not qevd_labels:
        reddit_issues = reddit_sample.get("targets_closed", {}).get("issues_present", [])
        qevd_labels = []
        for issue in reddit_issues:
            mapped = REDDIT_TO_QEVD.get(issue)
            if mapped and mapped not in qevd_labels:
                qevd_labels.append(mapped)

    structured = qevd_labels_to_structured(qevd_labels)

    # Build the sample in QEVD eval format
    return {
        "id": f"reddit_{sid}",
        "video": f"reddit_videos/{video_name}",
        "conversations": [
            {"from": "human", "value": CLASSIFICATION_PROMPT.strip()},
            {"from": "gpt", "value": ""},  # No reference response for eval
        ],
        "metadata": {
            "structured_labels": structured,
            "original_flat_labels": qevd_labels,
            "split": "test",
            "domain": "wild",
            "source": "reddit",
            "reddit_id": sid,
            "exercise": reddit_sample.get("metadata", {}).get("exercise", "unknown"),
            "camera_view": reddit_sample.get("metadata", {}).get("view", "unknown"),
            "reddit_issues": reddit_sample.get("targets_closed", {}).get("issues_present", []),
            "reddit_form_score": reddit_sample.get("targets", {}).get("form_score"),
            "mapping_notes": reddit_sample.get("qevd_mapping_notes", {}),
        },
    }


def main():
    # Create symlink for video access
    if not VIDEO_LINK_DIR.exists():
        VIDEO_LINK_DIR.symlink_to(VIDEO_DIR.resolve())
        print(f"Created symlink: {VIDEO_LINK_DIR} → {VIDEO_DIR.resolve()}")

    # Load Reddit data
    reddit_data = []
    with open(INPUT_JSONL) as f:
        for line in f:
            if line.strip():
                reddit_data.append(json.loads(line))
    print(f"Loaded {len(reddit_data)} Reddit samples")

    # Convert
    converted = []
    skipped = 0
    for sample in reddit_data:
        out = convert_reddit_sample(sample)
        # Check video exists
        video_path = VIDEO_DIR / sample.get("video_path", "")
        if not video_path.exists():
            print(f"  SKIP (video missing): {sample.get('video_path')}")
            skipped += 1
            continue
        converted.append(out)

    print(f"Converted {len(converted)} samples ({skipped} skipped)")

    # Label distribution
    label_counts = Counter()
    for s in converted:
        for l in s["metadata"]["original_flat_labels"]:
            clean = l.replace("squats - ", "")
            label_counts[clean] += 1

    print("\nReddit label distribution:")
    for l, c in label_counts.most_common():
        print(f"  {l}: {c}")

    # Check which labels have no samples
    all_labels = ["shoulder-width", "narrow", "wide", "plie", "shallow",
                  "90 degrees", "over 90 degrees", "back not straight",
                  "knees over toes", "insufficient", "hold", "not visible"]
    missing = [l for l in all_labels if label_counts.get(l, 0) == 0]
    if missing:
        print(f"\nLabels with NO Reddit samples: {missing}")
        print("Transfer gap analysis will be limited to labels present in both domains.")

    # Camera view distribution
    views = Counter(s["metadata"]["camera_view"] for s in converted)
    print(f"\nCamera views: {dict(views)}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH} ({len(converted)} samples)")


if __name__ == "__main__":
    main()
