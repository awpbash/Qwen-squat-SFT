"""
v2: Unified evaluation for squat VLM (classification metrics).

Changes from v1:
  - Built-in bootstrap 95% CI for all metrics
  - --domain flag for controlled/wild evaluation
  - Cleaner metric summary with confidence intervals
  - Saves both metrics.json and predictions.jsonl

Usage:
  # QEVD controlled
  python eval_unified.py --model_path <path> --data_path new_qevd_mrq_test.json \\
      --output_dir eval_results/E2_sft_lora/controlled/

  # Reddit wild
  python eval_unified.py --model_path <path> --data_path reddit_eval_test.json \\
      --output_dir eval_results/E2_sft_lora/wild/ --domain wild
"""

import argparse
import json
import re
import csv
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# ── Label definitions (shared) ─────────────────────────────────────
from constants import (
    STANCE_LABELS, DEPTH_LABELS, FORM_LABELS, ALL_LABELS,
    GEOMETRIC_LABELS, HOLISTIC_LABELS, TEMPORAL_LABELS,
)

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 123


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--video_root", default=".")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--domain", choices=["controlled", "wild"], default="controlled")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--video_min_pixels", type=int, default=None)
    p.add_argument("--video_max_pixels", type=int, default=None)
    p.add_argument("--skip_inference", action="store_true",
                   help="Recompute metrics from existing predictions.jsonl")
    return p.parse_args()


def load_model(model_path: str, base_model: Optional[str], device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    adapter_path = Path(model_path) / "adapter_model.safetensors"
    is_lora = adapter_path.exists()

    if is_lora and base_model:
        print(f"Loading base model: {base_model}")
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            base_model, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter: {model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

        # Load non-LoRA trained weights (e.g., unfrozen merger)
        non_lora_path = Path(model_path) / "non_lora_state_dict.bin"
        if non_lora_path.exists():
            import torch as _torch
            print(f"Loading non-LoRA state dict: {non_lora_path}")
            state_dict = _torch.load(str(non_lora_path), map_location="cpu")
            # Strip 'base_model.model.' prefix added by PEFT during training
            cleaned = {}
            for k, v in state_dict.items():
                clean_key = k.replace("base_model.model.", "", 1)
                cleaned[clean_key] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"  Loaded {len(cleaned)} non-LoRA params (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"Loading model: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

    model.eval()
    return model, processor


def extract_structured_from_text(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if "visible" in obj or "stance" in obj or "labels" in obj:
            return obj
        return None
    except (json.JSONDecodeError, AttributeError):
        return None


def structured_to_label_set(obj: dict) -> set:
    if obj is None:
        return set()

    if "labels" in obj:
        return {l.replace("squats - ", "") for l in obj["labels"]}

    labels = set()
    if obj.get("visible") is False:
        return {"not visible"}

    _ALL_SET = set(ALL_LABELS)
    stance = str(obj.get("stance") or "").lower()
    if stance in _ALL_SET:
        labels.add(stance)
    depth = str(obj.get("depth") or "").lower()
    if depth in _ALL_SET:
        labels.add(depth)
    form_issues = obj.get("form_issues", [])
    if isinstance(form_issues, list):
        for issue in form_issues:
            normalized = str(issue).lower() if issue else ""
            if normalized in _ALL_SET:
                labels.add(normalized)
    variant = str(obj.get("variant") or "").lower()
    if variant in _ALL_SET:
        labels.add(variant)

    return labels


def get_true_labels(sample: dict) -> set:
    meta = sample.get("metadata", {})
    structured = meta.get("structured_labels")
    if structured:
        return structured_to_label_set(structured)
    flat = meta.get("original_flat_labels", [])
    if flat:
        return {l.replace("squats - ", "") for l in flat}
    return set()


@torch.no_grad()
def predict(model, processor, video_path: str, prompt: str, fps: float, device: str,
            video_min_pixels=None, video_max_pixels=None) -> tuple:
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

    gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
    gen = gen_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

    parsed = extract_structured_from_text(raw)
    pred_labels = structured_to_label_set(parsed)
    return pred_labels, raw, parsed


def compute_metrics(y_true_list: list[set], y_pred_list: list[set]) -> dict:
    per_label = {}
    for label in ALL_LABELS:
        tp = sum(1 for t, p in zip(y_true_list, y_pred_list) if label in t and label in p)
        fp = sum(1 for t, p in zip(y_true_list, y_pred_list) if label not in t and label in p)
        fn = sum(1 for t, p in zip(y_true_list, y_pred_list) if label in t and label not in p)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "f1": f1}

    total_tp = sum(v["TP"] for v in per_label.values())
    total_fp = sum(v["FP"] for v in per_label.values())
    total_fn = sum(v["FN"] for v in per_label.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
    macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)

    # macro_f1_active: average F1 only over labels with ground-truth support > 0.
    # Prevents zero-support labels (e.g. labels absent from Reddit wild data)
    # from artificially deflating the macro average.
    active_f1s = [v["f1"] for v in per_label.values() if (v["TP"] + v["FN"]) > 0]
    macro_f1_active = sum(active_f1s) / len(active_f1s) if active_f1s else 0.0
    n_active_labels = len(active_f1s)

    n = len(y_true_list) or 1
    subset_acc = sum(1 for t, p in zip(y_true_list, y_pred_list) if t == p) / n
    hamming = sum(
        sum(1 for l in ALL_LABELS if (l in t) != (l in p)) / len(ALL_LABELS)
        for t, p in zip(y_true_list, y_pred_list)
    ) / n

    stance_correct = sum(1 for t, p in zip(y_true_list, y_pred_list)
                         if (t & set(STANCE_LABELS)) == (p & set(STANCE_LABELS)))
    depth_correct = sum(1 for t, p in zip(y_true_list, y_pred_list)
                        if (t & set(DEPTH_LABELS)) == (p & set(DEPTH_LABELS)))
    stance_acc = stance_correct / n
    depth_acc = depth_correct / n

    labels_with_recall = sum(1 for v in per_label.values() if v["recall"] > 0)
    label_coverage = labels_with_recall / len(ALL_LABELS)

    # Per-group F1 averages
    geo_f1s = [per_label[l]["f1"] for l in ALL_LABELS if l in GEOMETRIC_LABELS]
    hol_f1s = [per_label[l]["f1"] for l in ALL_LABELS if l in HOLISTIC_LABELS]
    geo_avg_f1 = sum(geo_f1s) / len(geo_f1s) if geo_f1s else 0.0
    hol_avg_f1 = sum(hol_f1s) / len(hol_f1s) if hol_f1s else 0.0

    return {
        "per_label": per_label,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "TP": total_tp, "FP": total_fp, "FN": total_fn},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "macro_f1_active": macro_f1_active,
        "n_active_labels": n_active_labels,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "stance_accuracy": stance_acc,
        "depth_accuracy": depth_acc,
        "label_coverage": label_coverage,
        "geometric_avg_f1": geo_avg_f1,
        "holistic_avg_f1": hol_avg_f1,
    }


def bootstrap_ci(y_true_list, y_pred_list, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Compute bootstrap 95% CI for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true_list)
    boot_metrics = {k: [] for k in [
        "macro_f1", "macro_f1_active", "micro_f1", "subset_accuracy", "hamming_loss",
        "stance_accuracy", "depth_accuracy", "geometric_avg_f1", "holistic_avg_f1"
    ]}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bt = [y_true_list[i] for i in idx]
        bp = [y_pred_list[i] for i in idx]
        m = compute_metrics(bt, bp)
        boot_metrics["macro_f1"].append(m["macro"]["f1"])
        boot_metrics["macro_f1_active"].append(m["macro_f1_active"])
        boot_metrics["micro_f1"].append(m["micro"]["f1"])
        boot_metrics["subset_accuracy"].append(m["subset_accuracy"])
        boot_metrics["hamming_loss"].append(m["hamming_loss"])
        boot_metrics["stance_accuracy"].append(m["stance_accuracy"])
        boot_metrics["depth_accuracy"].append(m["depth_accuracy"])
        boot_metrics["geometric_avg_f1"].append(m["geometric_avg_f1"])
        boot_metrics["holistic_avg_f1"].append(m["holistic_avg_f1"])

    ci = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "se": float(np.std(arr)),
        }
    return ci


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]

    y_true_list = []
    y_pred_list = []
    predictions = []
    parse_failures = 0

    pred_path = out_dir / "predictions.jsonl"

    if args.skip_inference and pred_path.exists():
        # Recompute metrics from existing predictions
        print(f"Loading existing predictions from {pred_path}")
        with open(pred_path) as f:
            for line in f:
                rec = json.loads(line)
                if "error" in rec and "true_labels" not in rec:
                    continue
                y_true_list.append(set(rec.get("true_labels", [])))
                y_pred_list.append(set(rec.get("pred_labels", [])))
                predictions.append(rec)
    else:
        # Run inference
        model, processor = load_model(args.model_path, args.base_model, device)
        prompt = data[0]["conversations"][0]["value"]
        print(f"Evaluating {len(data)} samples [{args.domain}] at fps={args.fps}")

        for sample in tqdm(data, desc=f"Eval [{args.domain}]"):
            sid = sample.get("id", "")
            video_rel = sample.get("video", "")
            video_abs = str((Path(args.video_root) / video_rel).resolve())

            if not Path(video_abs).exists():
                predictions.append({"id": sid, "error": "video_missing", "video": video_rel})
                continue

            true_labels = get_true_labels(sample)

            try:
                pred_labels, raw, parsed = predict(
                    model, processor, video_abs, prompt, args.fps, device,
                    video_min_pixels=args.video_min_pixels,
                    video_max_pixels=args.video_max_pixels,
                )
            except Exception as e:
                predictions.append({"id": sid, "error": str(e), "video": video_rel})
                pred_labels = set()
                raw = ""
                parsed = None

            if parsed is None:
                parse_failures += 1

            y_true_list.append(true_labels)
            y_pred_list.append(pred_labels)
            predictions.append({
                "id": sid,
                "video": video_rel,
                "true_labels": sorted(true_labels),
                "pred_labels": sorted(pred_labels),
                "prediction_raw": raw,
                "domain": args.domain,
            })

        # Save predictions
        with open(pred_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Compute metrics
    metrics = compute_metrics(y_true_list, y_pred_list)
    metrics["domain"] = args.domain
    metrics["total_samples"] = len(y_true_list)
    metrics["parse_failures"] = parse_failures
    metrics["json_parse_rate"] = 1.0 - (parse_failures / len(y_true_list)) if y_true_list else 0.0

    # Bootstrap CI
    print("Computing bootstrap 95% CI...")
    metrics["bootstrap_ci"] = bootstrap_ci(y_true_list, y_pred_list)

    # Save
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
            elif label in TEMPORAL_LABELS:
                group = "temporal"
            else:
                group = "meta"
            writer.writerow([label, group, m["TP"], m["FP"], m["FN"],
                           f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])

    # Print summary
    ci = metrics["bootstrap_ci"]
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS — {args.domain.upper()} ({len(y_true_list)} samples)")
    print(f"{'='*70}")
    print(f"Macro F1:        {metrics['macro']['f1']:.4f}  [{ci['macro_f1']['ci_low']:.3f}, {ci['macro_f1']['ci_high']:.3f}]")
    print(f"Macro F1 (act.): {metrics['macro_f1_active']:.4f}  [{ci['macro_f1_active']['ci_low']:.3f}, {ci['macro_f1_active']['ci_high']:.3f}]  ({metrics['n_active_labels']}/{len(ALL_LABELS)} labels with support)")
    print(f"Micro F1:        {metrics['micro']['f1']:.4f}  [{ci['micro_f1']['ci_low']:.3f}, {ci['micro_f1']['ci_high']:.3f}]")
    print(f"Subset Acc:      {metrics['subset_accuracy']:.4f}  [{ci['subset_accuracy']['ci_low']:.3f}, {ci['subset_accuracy']['ci_high']:.3f}]")
    print(f"Hamming Loss:    {metrics['hamming_loss']:.4f}  [{ci['hamming_loss']['ci_low']:.3f}, {ci['hamming_loss']['ci_high']:.3f}]")
    print(f"Stance Acc:      {metrics['stance_accuracy']:.4f}  [{ci['stance_accuracy']['ci_low']:.3f}, {ci['stance_accuracy']['ci_high']:.3f}]")
    print(f"Depth Acc:       {metrics['depth_accuracy']:.4f}  [{ci['depth_accuracy']['ci_low']:.3f}, {ci['depth_accuracy']['ci_high']:.3f}]")
    print(f"Geometric F1:    {metrics['geometric_avg_f1']:.4f}  [{ci['geometric_avg_f1']['ci_low']:.3f}, {ci['geometric_avg_f1']['ci_high']:.3f}]")
    print(f"Holistic F1:     {metrics['holistic_avg_f1']:.4f}  [{ci['holistic_avg_f1']['ci_low']:.3f}, {ci['holistic_avg_f1']['ci_high']:.3f}]")
    print(f"Coverage:        {metrics['label_coverage']:.0%} ({int(metrics['label_coverage']*len(ALL_LABELS))}/{len(ALL_LABELS)})")
    print(f"JSON Parse Rate: {metrics['json_parse_rate']:.0%}")

    print(f"\n{'Label':<25} {'Grp':<5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 70)
    for label in ALL_LABELS:
        m = metrics["per_label"][label]
        grp = "G" if label in GEOMETRIC_LABELS else ("H" if label in HOLISTIC_LABELS else "T")
        flag = " !" if m["recall"] == 0 and (m["TP"] + m["FN"]) > 0 else ""
        print(f"{label:<25} {grp:<5} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}{flag}")

    print(f"\nSaved to: {out_dir}/")


if __name__ == "__main__":
    main()
