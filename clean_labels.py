"""
Phase 1: Clean label contradictions in QEVD squat training data.

Rules applied:
1. "not visible" is exclusive — remove ALL other labels when present
2. "plie" implies wide stance — remove "wide" and "shoulder-width" when plie present
3. Multi-stance conflicts — keep the non-default (non-shoulder-width) label
4. Multi-depth conflicts — keep the more specific (non-90-degrees) label
5. Remove "starting late" samples entirely (no visual grounding, video editing artifact)

Input:  new_qevd_mrq.json (4,308 samples)
Output: new_qevd_mrq_clean.json
Log:    label_cleaning_log.jsonl
"""

import json
import re
from pathlib import Path
from copy import deepcopy

INPUT_PATH = Path("new_qevd_mrq.json")
OUTPUT_PATH = Path("new_qevd_mrq_clean.json")
LOG_PATH = Path("label_cleaning_log.jsonl")

STANCE_LABELS = {"squats - shoulder-width", "squats - narrow", "squats - wide", "squats - plie"}
DEPTH_LABELS = {"squats - shallow", "squats - 90 degrees", "squats - over 90 degrees"}


def extract_labels(gpt_response: str) -> list[str]:
    m = re.search(r"\{.*\}", gpt_response, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj.get("labels", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def clean_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """Apply cleaning rules. Returns (cleaned_labels, list_of_changes)."""
    original = set(labels)
    cleaned = set(labels)
    changes = []

    # Rule 1: "not visible" is exclusive
    if "squats - not visible" in cleaned:
        others = cleaned - {"squats - not visible"}
        if others:
            changes.append(f"not_visible_exclusive: removed {others}")
            cleaned = {"squats - not visible"}

    # Rule 5: Remove "starting late" (handled at sample level, but also strip from labels)
    if "squats - starting late" in cleaned:
        changes.append("removed_starting_late")
        cleaned.discard("squats - starting late")

    # Rule 2: Plie implies wide — remove "wide", "shoulder-width", AND "narrow"
    # Plie is a specific stance that overrides all other stance labels
    if "squats - plie" in cleaned:
        for remove_label in ["squats - wide", "squats - shoulder-width", "squats - narrow"]:
            if remove_label in cleaned:
                changes.append(f"plie_overrides_stance: removed '{remove_label}'")
                cleaned.discard(remove_label)

    # Rule 3: Multi-stance conflicts — keep non-default
    stance_present = cleaned & {"squats - shoulder-width", "squats - narrow", "squats - wide"}
    if len(stance_present) > 1:
        # Priority: narrow > wide > shoulder-width (keep most specific/unusual)
        if "squats - shoulder-width" in stance_present:
            changes.append(f"multi_stance: removed 'shoulder-width', kept {stance_present - {'squats - shoulder-width'}}")
            cleaned.discard("squats - shoulder-width")
        # If still >1 (narrow + wide), keep narrow
        stance_present = cleaned & {"squats - narrow", "squats - wide"}
        if len(stance_present) > 1:
            changes.append(f"multi_stance: removed 'wide', kept 'narrow'")
            cleaned.discard("squats - wide")

    # Rule 4: Multi-depth — keep only one depth label
    depth_present = cleaned & DEPTH_LABELS
    if len(depth_present) > 1:
        # Priority: remove "90 degrees" first, then "shallow" (keep most extreme)
        if "squats - 90 degrees" in depth_present:
            changes.append(f"multi_depth: removed '90 degrees', kept {depth_present - {'squats - 90 degrees'}}")
            cleaned.discard("squats - 90 degrees")
        depth_present = cleaned & DEPTH_LABELS
        if len(depth_present) > 1:
            # shallow + over 90 degrees: keep "over 90 degrees" (more specific)
            if "squats - shallow" in depth_present:
                changes.append(f"multi_depth: removed 'shallow', kept {depth_present - {'squats - shallow'}}")
                cleaned.discard("squats - shallow")

    return sorted(cleaned), changes


def rebuild_gpt_response(original_response: str, new_labels: list[str], descriptive_labels: list[str]) -> str:
    """Rebuild the GPT response with cleaned labels."""
    # Build visual analysis from descriptive labels
    analysis_lines = [f"- {desc}" for desc in descriptive_labels]
    analysis = "\n".join(analysis_lines)

    labels_json = json.dumps({"labels": new_labels})

    return (
        f"**Visual Analysis:**\n{analysis}\n\n"
        f"**JSON Classification:**\n```json\n{labels_json}\n```"
    )


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {INPUT_PATH}")

    cleaned_data = []
    log_entries = []
    dropped_starting_late = 0
    samples_modified = 0
    total_changes = 0

    for sample in data:
        sid = sample.get("id", "unknown")
        conv = sample.get("conversations", [])
        if len(conv) < 2:
            continue

        gpt_response = conv[1].get("value", "")
        labels = extract_labels(gpt_response)

        # Drop "starting late" only samples entirely
        if labels == ["squats - starting late"]:
            dropped_starting_late += 1
            log_entries.append({
                "id": sid,
                "action": "dropped_sample",
                "reason": "starting_late_only",
                "original_labels": labels,
            })
            continue

        new_labels, changes = clean_labels(labels)

        if not new_labels:
            # If all labels were removed (shouldn't happen), skip
            log_entries.append({
                "id": sid,
                "action": "dropped_sample",
                "reason": "no_labels_after_cleaning",
                "original_labels": labels,
            })
            continue

        new_sample = deepcopy(sample)

        if changes:
            samples_modified += 1
            total_changes += len(changes)

            # Update descriptive labels in metadata to match cleaned labels
            meta = new_sample.get("metadata", {})
            orig_descriptive = meta.get("labels_descriptive", [])

            # Filter descriptive labels to only keep ones matching cleaned labels
            label_to_desc = {}
            for desc in orig_descriptive:
                for label in new_labels:
                    # Match descriptive to label by checking if the label prefix appears
                    label_prefix = label.replace("squats - ", "squats - ")
                    if desc.lower().startswith(label_prefix.split(" - ")[0].lower()):
                        label_to_desc[label] = desc

            # Keep descriptive labels that correspond to remaining labels
            new_descriptive = []
            for label in new_labels:
                if label in label_to_desc:
                    new_descriptive.append(label_to_desc[label])
                else:
                    # Fallback: use the label itself as description
                    new_descriptive.append(label)

            meta["labels_descriptive"] = new_descriptive
            new_sample["metadata"] = meta

            # Rebuild GPT response
            new_sample["conversations"][1]["value"] = rebuild_gpt_response(
                gpt_response, new_labels, new_descriptive
            )

            log_entries.append({
                "id": sid,
                "action": "modified",
                "changes": changes,
                "original_labels": labels,
                "cleaned_labels": new_labels,
            })

        cleaned_data.append(new_sample)

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    with open(LOG_PATH, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n=== Cleaning Summary ===")
    print(f"Input samples:       {len(data)}")
    print(f"Dropped (starting late): {dropped_starting_late}")
    print(f"Output samples:      {len(cleaned_data)}")
    print(f"Samples modified:    {samples_modified}")
    print(f"Total label changes: {total_changes}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Log:      {LOG_PATH}")


if __name__ == "__main__":
    main()
