"""
v3: Generate thesis-quality tables (LaTeX + Markdown) and figures from experiment results.

Generates:
  Tables (LaTeX + Markdown):
    - Table 2: Main results (controlled + wild) with CIs
    - Table 3: Per-label F1 decomposition (geometric vs holistic)
    - Table 4: Transfer gap analysis
    - Table 5: Agentic calibration analysis
    - Table 6: Coaching quality metrics
    - Table 7: Pairwise McNemar's significance tests

  Figures (PDF + PNG):
    - Fig 2: Label distribution histogram (training data)
    - Fig 3: Per-label F1 grouped bar chart (controlled vs wild)
    - Fig 4: Transfer gap waterfall chart
    - Fig 5: Calibration scatter (E5 vs E5b)
    - Fig 6: F1 heatmap (experiments x labels)
    - Fig 7: Radar chart comparing models

Usage:
  python thesis_tables.py --experiments_dir eval_results/ --output_dir thesis_figures/
"""

import argparse
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from constants import (
    STANCE_LABELS, DEPTH_LABELS, FORM_LABELS, ALL_LABELS,
    GEOMETRIC_LABELS, HOLISTIC_LABELS, TEMPORAL_LABELS,
)

# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    "E1_zero_shot": "#E69F00",          # Orange
    "E2_sft": "#56B4E9",                # Sky blue
    "E3_pose_only": "#009E73",          # Bluish green
    "E4_always_tool": "#F0E442",        # Yellow
    "E5_agentic_untrained": "#0072B2",  # Blue
    "E5b_agentic_trained": "#D55E00",   # Vermilion
    "E5c_agentic_sft": "#CC79A7",      # Reddish purple (Okabe-Ito)
}
GEOMETRIC_COLOR = "#0072B2"
HOLISTIC_COLOR = "#D55E00"
TEMPORAL_COLOR = "#009E73"

# Short display names for figures
DISPLAY_NAMES = {
    "E1_zero_shot": "E1: Zero-shot",
    "E2_sft": "E2: SFT",
    "E3_pose_only": "E3: Pose-only",
    "E4_always_tool": "E4: Always-tool",
    "E5_agentic_untrained": "E5: Agentic",
    "E5b_agentic_trained": "E5b: Agentic-FT",
    "E5c_agentic_sft": "E5c: SFT+Agentic-Inf",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", default="eval_results")
    p.add_argument("--output_dir", default="thesis_figures")
    p.add_argument("--train_data", default="new_qevd_mrq_train.json",
                   help="Path to training data JSON (for label distribution figure)")
    return p.parse_args()


def get_color(exp_name):
    """Get color for an experiment, with fallback."""
    return COLORS.get(exp_name, "#999999")


def get_display_name(exp_name):
    """Get short display name for an experiment."""
    return DISPLAY_NAMES.get(exp_name, exp_name)


def load_metrics(exp_dir: Path, exp_name: str, domain: str = "controlled") -> dict:
    """Try loading metrics from domain subdir or direct."""
    paths = [
        exp_dir / exp_name / domain / "metrics.json",
        exp_dir / exp_name / "metrics.json",
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


# ═══════════════════════════════════════════════════════════════════════
# TABLE 2: Main results
# ═══════════════════════════════════════════════════════════════════════
def generate_main_results_table(all_metrics: dict, out_dir: Path):
    """Table 2: Main results across all experiments and domains."""
    rows = []
    for (exp, domain), m in sorted(all_metrics.items()):
        ci = m.get("bootstrap_ci", {})
        rows.append({
            "exp": exp, "domain": domain,
            "macro_f1": m["macro"]["f1"],
            "macro_f1_active": m.get("macro_f1_active", m["macro"]["f1"]),
            "n_active": m.get("n_active_labels", len(ALL_LABELS)),
            "ci_low": ci.get("macro_f1_active", ci.get("macro_f1", {})).get("ci_low", 0),
            "ci_high": ci.get("macro_f1_active", ci.get("macro_f1", {})).get("ci_high", 0),
            "has_ci": bool(ci.get("macro_f1_active") or ci.get("macro_f1")),
            "micro_f1": m["micro"]["f1"],
            "subset_acc": m.get("subset_accuracy", 0),
            "hamming": m.get("hamming_loss", 0),
            "stance_acc": m.get("stance_accuracy", 0),
            "depth_acc": m.get("depth_accuracy", 0),
            "geo_f1": m.get("geometric_avg_f1", 0),
            "hol_f1": m.get("holistic_avg_f1", 0),
            "coverage": m.get("label_coverage", 0),
            "n": m.get("total_samples", 0),
        })

    # Markdown
    with open(out_dir / "table2_main_results.md", "w") as f:
        f.write("| Experiment | Domain | N | Macro F1 (active) | 95% CI | Micro F1 | Subset | Hamming | Stance | Depth | Geo F1 | Hol F1 | Cover |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r["has_ci"] else "---"
            f.write(f"| {r['exp']} | {r['domain']} | {r['n']} | "
                    f"**{r['macro_f1_active']:.3f}** ({r['n_active']}) | {ci_str} | {r['micro_f1']:.3f} | "
                    f"{r['subset_acc']:.3f} | {r['hamming']:.3f} | {r['stance_acc']:.3f} | "
                    f"{r['depth_acc']:.3f} | {r['geo_f1']:.3f} | {r['hol_f1']:.3f} | "
                    f"{r['coverage']:.0%} |\n")

    # LaTeX
    with open(out_dir / "table2_main_results.tex", "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Main results across all experiments and evaluation domains. "
                "Macro F1 (active) averages only over labels with ground-truth support $>0$, "
                "preventing zero-support labels from deflating wild-domain scores. "
                "Best results per domain are \\textbf{bolded}. CI = bootstrap 95\\% confidence interval (5000 resamples).}\n"
                "\\label{tab:main-results}\n"
                "\\small\n"
                "\\begin{tabular}{llrcccccccc}\n\\toprule\n"
                "Experiment & Domain & $N$ & Macro F1 (act.) & 95\\% CI & Micro F1 & Subset Acc & Hamming$\\downarrow$ & Geo F1 & Hol F1 & Cover \\\\\n"
                "\\midrule\n")

        # Find best macro F1 (active) per domain
        best = {}
        for r in rows:
            if r["domain"] not in best or r["macro_f1_active"] > best[r["domain"]]:
                best[r["domain"]] = r["macro_f1_active"]

        prev_domain = None
        for r in rows:
            if prev_domain and r["domain"] != prev_domain:
                f.write("\\midrule\n")
            prev_domain = r["domain"]

            ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r["has_ci"] else "---"
            macro = f"\\textbf{{{r['macro_f1_active']:.3f}}}" if r["macro_f1_active"] == best[r["domain"]] else f"{r['macro_f1_active']:.3f}"
            exp_name = escape_latex(get_display_name(r["exp"]))
            f.write(f"{exp_name} & {r['domain']} & {r['n']} & "
                    f"{macro} & {ci_str} & {r['micro_f1']:.3f} & "
                    f"{r['subset_acc']:.3f} & {r['hamming']:.3f} & "
                    f"{r['geo_f1']:.3f} & {r['hol_f1']:.3f} & "
                    f"{r['coverage']:.0%} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print("  Generated table2_main_results.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 3: Per-label F1
# ═══════════════════════════════════════════════════════════════════════
def generate_per_label_table(all_metrics: dict, out_dir: Path):
    """Table 3: Per-label F1 across experiments (controlled domain)."""
    controlled = {exp: m for (exp, domain), m in all_metrics.items() if domain == "controlled"}
    if not controlled:
        return

    exp_names = sorted(controlled.keys())

    # Markdown
    with open(out_dir / "table3_per_label.md", "w") as f:
        header = "| Label | Group |" + " | ".join(exp_names) + " |"
        f.write(header + "\n")
        f.write("|---|---|" + "|".join(["---"] * len(exp_names)) + "|\n")

        for label in ALL_LABELS:
            group = "Geo" if label in GEOMETRIC_LABELS else ("Hol" if label in HOLISTIC_LABELS else "Tmp")
            vals = []
            for exp in exp_names:
                f1 = controlled[exp]["per_label"].get(label, {}).get("f1", 0)
                vals.append(f"{f1:.3f}")
            f.write(f"| {label} | {group} | " + " | ".join(vals) + " |\n")

        for group_name, group_set in [("Avg Geometric", GEOMETRIC_LABELS), ("Avg Holistic", HOLISTIC_LABELS)]:
            vals = []
            for exp in exp_names:
                avg = np.mean([controlled[exp]["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS if l in group_set])
                vals.append(f"**{avg:.3f}**")
            f.write(f"| **{group_name}** | --- | " + " | ".join(vals) + " |\n")

    # LaTeX
    with open(out_dir / "table3_per_label.tex", "w") as f:
        ncols = len(exp_names)
        col_spec = "ll" + "c" * ncols
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Per-label F1 scores across experiments (controlled domain). "
                "Best per row is \\textbf{bolded}. Geo = geometric, Hol = holistic, Tmp = temporal.}\n"
                "\\label{tab:per-label}\n\\small\n"
                f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

        header_names = [escape_latex(get_display_name(e)) for e in exp_names]
        f.write("Label & Group & " + " & ".join(header_names) + " \\\\\n\\midrule\n")

        for label in ALL_LABELS:
            group = "Geo" if label in GEOMETRIC_LABELS else ("Hol" if label in HOLISTIC_LABELS else "Tmp")
            f1_vals = [controlled[exp]["per_label"].get(label, {}).get("f1", 0) for exp in exp_names]
            best_val = max(f1_vals)
            cells = []
            for v in f1_vals:
                s = f"{v:.3f}"
                if v == best_val and best_val > 0:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            f.write(f"{escape_latex(label)} & {group} & " + " & ".join(cells) + " \\\\\n")

        f.write("\\midrule\n")
        for group_name, group_set in [("Avg Geometric", GEOMETRIC_LABELS), ("Avg Holistic", HOLISTIC_LABELS)]:
            avgs = [np.mean([controlled[exp]["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS if l in group_set]) for exp in exp_names]
            best_avg = max(avgs)
            cells = [f"\\textbf{{{v:.3f}}}" if v == best_avg else f"{v:.3f}" for v in avgs]
            f.write(f"\\textit{{{group_name}}} & --- & " + " & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print("  Generated table3_per_label.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 4: Transfer gap analysis
# ═══════════════════════════════════════════════════════════════════════
def generate_transfer_gap_table(all_metrics: dict, out_dir: Path):
    """Table 4: Transfer gap (controlled - wild) per experiment."""
    paired = {}
    for (exp, domain), m in all_metrics.items():
        if exp not in paired:
            paired[exp] = {}
        paired[exp][domain] = m

    experiments_with_both = {k: v for k, v in paired.items() if "controlled" in v and "wild" in v}
    if not experiments_with_both:
        print("  Skipping table4 (no paired controlled/wild data)")
        return

    exp_names = sorted(experiments_with_both.keys())

    # LaTeX
    with open(out_dir / "table4_transfer_gap.tex", "w") as f:
        ncols = len(exp_names)
        col_spec = "l" + "ccc" * ncols
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Transfer gap analysis: F1 scores on controlled (C) vs.\\ wild (W) domains, "
                "and gap $\\Delta$ = C $-$ W. Positive $\\Delta$ indicates degradation on wild data.}\n"
                "\\label{tab:transfer-gap}\n\\small\n"
                f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

        # Header row 1: experiment names
        header1 = " & " + " & ".join([f"\\multicolumn{{3}}{{c}}{{{escape_latex(get_display_name(e))}}}" for e in exp_names])
        f.write(header1 + " \\\\\n")
        for i, e in enumerate(exp_names):
            start = 2 + i * 3
            end = start + 2
            f.write(f"\\cmidrule(lr){{{start}-{end}}}")
        f.write("\n")

        # Header row 2: C / W / Delta
        header2 = "Label" + " & C & W & $\\Delta$" * ncols
        f.write(header2 + " \\\\\n\\midrule\n")

        for label in ALL_LABELS:
            cells = [escape_latex(label)]
            for exp in exp_names:
                ctrl_f1 = experiments_with_both[exp]["controlled"]["per_label"].get(label, {}).get("f1", 0)
                wild_f1 = experiments_with_both[exp]["wild"]["per_label"].get(label, {}).get("f1", 0)
                gap = ctrl_f1 - wild_f1
                gap_str = f"{gap:+.3f}"
                if gap > 0.1:
                    gap_str = f"\\textcolor{{red}}{{{gap_str}}}"
                elif gap < -0.1:
                    gap_str = f"\\textcolor{{blue}}{{{gap_str}}}"
                cells.extend([f"{ctrl_f1:.3f}", f"{wild_f1:.3f}", gap_str])
            f.write(" & ".join(cells) + " \\\\\n")

        # Averages
        f.write("\\midrule\n")
        for group_name, group_set in [("Geometric Avg", GEOMETRIC_LABELS), ("Holistic Avg", HOLISTIC_LABELS), ("Macro Avg (active)", None)]:
            cells = [f"\\textit{{{group_name}}}"]
            for exp in exp_names:
                ctrl = experiments_with_both[exp]["controlled"]
                wild = experiments_with_both[exp]["wild"]
                if group_set is None:
                    # Use macro_f1_active (only labels with support>0) to avoid
                    # zero-support labels artificially deflating wild scores.
                    ctrl_f1 = ctrl.get("macro_f1_active", ctrl["macro"]["f1"])
                    wild_f1 = wild.get("macro_f1_active", wild["macro"]["f1"])
                else:
                    ctrl_f1 = np.mean([ctrl["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS if l in group_set])
                    wild_f1 = np.mean([wild["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS if l in group_set])
                gap = ctrl_f1 - wild_f1
                gap_str = f"{gap:+.3f}"
                cells.extend([f"\\textbf{{{ctrl_f1:.3f}}}", f"\\textbf{{{wild_f1:.3f}}}", f"\\textbf{{{gap_str}}}"])
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # Markdown
    with open(out_dir / "table4_transfer_gap.md", "w") as f:
        header = "| Label |" + " | ".join([f"{e} C | {e} W | {e} Gap" for e in exp_names]) + " |\n"
        f.write(header)
        f.write("|---|" + "---|---|---|" * len(exp_names) + "\n")
        for label in ALL_LABELS:
            row = f"| {label} |"
            for exp in exp_names:
                ctrl_f1 = experiments_with_both[exp]["controlled"]["per_label"].get(label, {}).get("f1", 0)
                wild_f1 = experiments_with_both[exp]["wild"]["per_label"].get(label, {}).get("f1", 0)
                gap = ctrl_f1 - wild_f1
                row += f" {ctrl_f1:.3f} | {wild_f1:.3f} | {gap:+.3f} |"
            f.write(row + "\n")

    print("  Generated table4_transfer_gap.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 5: Agentic calibration
# ═══════════════════════════════════════════════════════════════════════
def generate_calibration_table(all_metrics: dict, out_dir: Path):
    """Table 5: Agentic calibration — tool invocation rates and accuracy."""
    agentic_exps = {}
    for (exp, domain), m in all_metrics.items():
        if "agentic" in exp.lower() and "calibration" in m:
            key = f"{exp}/{domain}"
            agentic_exps[key] = m

    if not agentic_exps:
        print("  Skipping table5 (no agentic experiments with calibration data)")
        return

    exp_names = sorted(agentic_exps.keys())

    # LaTeX
    with open(out_dir / "table5_calibration.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Agentic calibration analysis. "
                "Tool invocation rate, accuracy when tool is not invoked, and missed opportunity rate "
                "(cases where the tool would have helped but was not invoked).}\n"
                "\\label{tab:calibration}\n\\small\n"
                "\\begin{tabular}{llccc}\n\\toprule\n"
                "Experiment & Field & Invoke \\% & Acc (no tool) & Missed \\% \\\\\n"
                "\\midrule\n")

        for exp_name in exp_names:
            m = agentic_exps[exp_name]
            cal = m.get("calibration", {})
            first = True
            for group, data in sorted(cal.items()):
                exp_col = escape_latex(exp_name) if first else ""
                first = False
                invoke = data.get("tool_invocation_rate", 0) * 100
                acc = data.get("accuracy_when_not_invoked", 0) * 100
                missed = data.get("missed_opportunity_rate", 0) * 100
                f.write(f"{exp_col} & {escape_latex(group)} & {invoke:.1f}\\% & {acc:.1f}\\% & {missed:.1f}\\% \\\\\n")
            f.write("\\midrule\n")

        # Overall tool invocation rate
        f.write("\\midrule\n")
        for exp_name in exp_names:
            m = agentic_exps[exp_name]
            rate = m.get("tool_invocation_rate", 0) * 100
            macro_f1 = m["macro"]["f1"]
            f.write(f"\\textit{{{escape_latex(exp_name)}}} & \\textit{{Overall}} & "
                    f"{rate:.1f}\\% & \\multicolumn{{2}}{{c}}{{Macro F1 = {macro_f1:.3f}}} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Markdown
    with open(out_dir / "table5_calibration.md", "w") as f:
        f.write("| Experiment | Field | Invoke % | Acc (no tool) | Missed % |\n")
        f.write("|---|---|---|---|---|\n")
        for exp_name in exp_names:
            m = agentic_exps[exp_name]
            cal = m.get("calibration", {})
            for group, data in sorted(cal.items()):
                invoke = data.get("tool_invocation_rate", 0) * 100
                acc = data.get("accuracy_when_not_invoked", 0) * 100
                missed = data.get("missed_opportunity_rate", 0) * 100
                f.write(f"| {exp_name} | {group} | {invoke:.1f}% | {acc:.1f}% | {missed:.1f}% |\n")

    print("  Generated table5_calibration.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 6: Coaching quality
# ═══════════════════════════════════════════════════════════════════════
def generate_coaching_table(all_metrics: dict, exp_dir: Path, out_dir: Path):
    """Table 6: Coaching quality metrics across experiments."""
    coaching_data = {}
    for (exp, domain), m in all_metrics.items():
        # Look for coaching metrics in the experiment directory
        coaching_path = exp_dir / exp / domain / "coaching_metrics" / "coaching_metrics.json"
        if not coaching_path.exists():
            coaching_path = exp_dir / exp / "coaching_metrics" / "coaching_metrics.json"
        if coaching_path.exists():
            with open(coaching_path) as f:
                coaching_data[f"{exp}/{domain}"] = json.load(f)

    if not coaching_data:
        print("  Skipping table6 (no coaching metrics found)")
        return

    # LaTeX
    with open(out_dir / "table6_coaching.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Coaching quality metrics. "
                "Issue recall measures how many ground-truth issues are mentioned in coaching. "
                "Specificity is scored 0--2 (generic to actionable). Hallucination rate measures false issue mentions.}\n"
                "\\label{tab:coaching}\n\\small\n"
                "\\begin{tabular}{lccccc}\n\\toprule\n"
                "Experiment & Recall & Precision & Halluc.\\,$\\downarrow$ & Specificity & Contra.\\,$\\downarrow$ \\\\\n"
                "\\midrule\n")

        for name in sorted(coaching_data.keys()):
            d = coaching_data[name]
            recall = d.get("issue_recall", {}).get("mean", 0)
            precision = d.get("issue_precision", {}).get("mean", 0)
            halluc = d.get("hallucination_rate", {}).get("mean", 0)
            spec = d.get("specificity", {}).get("mean", 0)
            contra = d.get("contradiction_rate", {}).get("mean", 0)
            f.write(f"{escape_latex(name)} & {recall:.3f} & {precision:.3f} & "
                    f"{halluc:.3f} & {spec:.2f} & {contra:.3f} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Markdown
    with open(out_dir / "table6_coaching.md", "w") as f:
        f.write("| Experiment | Issue Recall | Issue Precision | Hallucination | Specificity | Contradiction |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in sorted(coaching_data.keys()):
            d = coaching_data[name]
            f.write(f"| {name} "
                    f"| {d.get('issue_recall', {}).get('mean', 0):.3f} "
                    f"| {d.get('issue_precision', {}).get('mean', 0):.3f} "
                    f"| {d.get('hallucination_rate', {}).get('mean', 0):.3f} "
                    f"| {d.get('specificity', {}).get('mean', 0):.2f} "
                    f"| {d.get('contradiction_rate', {}).get('mean', 0):.3f} |\n")

    print("  Generated table6_coaching.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 7: McNemar's pairwise tests
# ═══════════════════════════════════════════════════════════════════════
def generate_mcnemar_table(exp_dir: Path, out_dir: Path):
    """Table 7: Pairwise McNemar's significance tests."""
    mcnemar_path = exp_dir / "statistical_summary" / "pairwise_mcnemar.csv"
    if not mcnemar_path.exists():
        print("  Skipping table7 (no pairwise_mcnemar.csv)")
        return

    rows = []
    with open(mcnemar_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("  Skipping table7 (empty pairwise_mcnemar.csv)")
        return

    # LaTeX
    with open(out_dir / "table7_mcnemar.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Pairwise McNemar's tests (subset accuracy). "
                "$b$ = A wrong, B right; $c$ = A right, B wrong. "
                "Significance: $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
                "HB = significant after Holm--Bonferroni correction.}\n"
                "\\label{tab:mcnemar}\n\\small\n"
                "\\begin{tabular}{llccccl}\n\\toprule\n"
                "Experiment A & Experiment B & $b$ & $c$ & $\\chi^2$ & $p$ & Sig. \\\\\n"
                "\\midrule\n")

        for r in rows:
            p_val = float(r["p_value"])
            if p_val < 0.001:
                sig = "$^{***}$"
            elif p_val < 0.01:
                sig = "$^{**}$"
            elif p_val < 0.05:
                sig = "$^{*}$"
            else:
                sig = "n.s."
            if r.get("significant_corrected") == "yes":
                sig += " (HB)"

            exp_a = escape_latex(get_display_name(r["experiment_a"]))
            exp_b = escape_latex(get_display_name(r["experiment_b"]))
            f.write(f"{exp_a} & {exp_b} & {r['b']} & {r['c']} & "
                    f"{float(r['chi2']):.2f} & {p_val:.4f} & {sig} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Markdown
    with open(out_dir / "table7_mcnemar.md", "w") as f:
        f.write("| Experiment A | Experiment B | b | c | chi2 | p-value | Significant | HB Corrected |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['experiment_a']} | {r['experiment_b']} | {r['b']} | {r['c']} | "
                    f"{r['chi2']} | {r['p_value']} | {r['significant']} | {r.get('significant_corrected', 'no')} |\n")

    print("  Generated table7_mcnemar.md + .tex")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: Label distribution histogram
# ═══════════════════════════════════════════════════════════════════════
def generate_label_distribution(data_path: Path, out_dir: Path):
    """Fig 2: Label distribution histogram from training data."""
    if not data_path.exists():
        print("  Skipping fig2 (no training data)")
        return

    from collections import Counter
    with open(data_path) as f:
        data = json.load(f)

    counts = Counter()
    for sample in data:
        labels = sample.get("metadata", {}).get("original_flat_labels", [])
        for l in labels:
            clean = l.replace("squats - ", "")
            counts[clean] += 1

    labels_sorted = [l for l in ALL_LABELS if counts.get(l, 0) > 0]
    values = [counts.get(l, 0) for l in labels_sorted]
    colors = [GEOMETRIC_COLOR if l in GEOMETRIC_LABELS else
              (HOLISTIC_COLOR if l in HOLISTIC_LABELS else TEMPORAL_COLOR)
              for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(labels_sorted)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count")
    ax.set_title("Label Distribution (Training Set, After Oversampling)")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GEOMETRIC_COLOR, label="Geometric"),
        Patch(facecolor=HOLISTIC_COLOR, label="Holistic"),
        Patch(facecolor=TEMPORAL_COLOR, label="Temporal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_label_distribution.pdf")
    fig.savefig(out_dir / "fig2_label_distribution.png")
    plt.close(fig)
    print("  Generated fig2_label_distribution.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Per-label F1 grouped bar chart (controlled vs wild)
# ═══════════════════════════════════════════════════════════════════════
def generate_per_label_bar_chart(all_metrics: dict, out_dir: Path):
    """Fig 3: Per-label F1 grouped bar chart comparing controlled vs wild for each experiment."""
    paired = {}
    for (exp, domain), m in all_metrics.items():
        if exp not in paired:
            paired[exp] = {}
        paired[exp][domain] = m

    experiments_with_both = {k: v for k, v in paired.items() if "controlled" in v and "wild" in v}
    if not experiments_with_both:
        print("  Skipping fig3 (no paired data)")
        return

    n_exps = len(experiments_with_both)
    exp_names = sorted(experiments_with_both.keys())
    n_labels = len(ALL_LABELS)

    fig, ax = plt.subplots(figsize=(14, 5))

    bar_width = 0.35 / max(n_exps, 1)
    group_width = n_exps * bar_width * 2 + 0.15
    x = np.arange(n_labels) * group_width

    for i, exp in enumerate(exp_names):
        ctrl_f1s = [experiments_with_both[exp]["controlled"]["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS]
        wild_f1s = [experiments_with_both[exp]["wild"]["per_label"].get(l, {}).get("f1", 0) for l in ALL_LABELS]

        offset = (i - n_exps / 2 + 0.5) * bar_width * 2
        color = get_color(exp)

        ax.bar(x + offset - bar_width / 2, ctrl_f1s, bar_width, label=f"{get_display_name(exp)} (ctrl)",
               color=color, alpha=0.9, edgecolor="black", linewidth=0.3)
        ax.bar(x + offset + bar_width / 2, wild_f1s, bar_width, label=f"{get_display_name(exp)} (wild)",
               color=color, alpha=0.4, edgecolor="black", linewidth=0.3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Label F1: Controlled (solid) vs Wild (hatched)")

    # Add vertical separators between label groups
    geo_count = len([l for l in ALL_LABELS if l in GEOMETRIC_LABELS])
    hol_end = geo_count + len([l for l in ALL_LABELS if l in HOLISTIC_LABELS])
    ax.axvline(x=(geo_count - 0.5) * group_width, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=(hol_end - 0.5) * group_width, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.legend(fontsize=7, ncol=n_exps, loc="upper right", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_per_label_bar.pdf")
    fig.savefig(out_dir / "fig3_per_label_bar.png")
    plt.close(fig)
    print("  Generated fig3_per_label_bar.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Transfer gap waterfall
# ═══════════════════════════════════════════════════════════════════════
def generate_transfer_gap_figure(all_metrics: dict, out_dir: Path):
    """Fig 4: Transfer gap waterfall chart."""
    paired = {}
    for (exp, domain), m in all_metrics.items():
        if exp not in paired:
            paired[exp] = {}
        paired[exp][domain] = m

    experiments_with_both = {k: v for k, v in paired.items() if "controlled" in v and "wild" in v}
    if not experiments_with_both:
        print("  Skipping fig4 (no paired data)")
        return

    n_exps = len(experiments_with_both)
    fig, axes = plt.subplots(1, n_exps, figsize=(5 * n_exps, 6), sharey=True, squeeze=False)

    for idx, (exp_name, data) in enumerate(sorted(experiments_with_both.items())):
        ax = axes[0][idx]
        ctrl = data["controlled"]
        wild = data["wild"]

        labels_with_data = []
        gaps = []
        colors = []

        for label in ALL_LABELS:
            ctrl_f1 = ctrl["per_label"].get(label, {}).get("f1", 0)
            wild_f1 = wild["per_label"].get(label, {}).get("f1", 0)
            ctrl_support = ctrl["per_label"].get(label, {}).get("TP", 0) + ctrl["per_label"].get(label, {}).get("FN", 0)
            wild_support = wild["per_label"].get(label, {}).get("TP", 0) + wild["per_label"].get(label, {}).get("FN", 0)

            if ctrl_support > 0 or wild_support > 0:
                gap = ctrl_f1 - wild_f1
                labels_with_data.append(label)
                gaps.append(gap)
                if label in GEOMETRIC_LABELS:
                    colors.append(GEOMETRIC_COLOR)
                elif label in HOLISTIC_LABELS:
                    colors.append(HOLISTIC_COLOR)
                else:
                    colors.append(TEMPORAL_COLOR)

        sorted_idx = np.argsort(gaps)[::-1]
        labels_sorted = [labels_with_data[i] for i in sorted_idx]
        gaps_sorted = [gaps[i] for i in sorted_idx]
        colors_sorted = [colors[i] for i in sorted_idx]

        y_pos = np.arange(len(labels_sorted))
        ax.barh(y_pos, gaps_sorted, color=colors_sorted, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel(r"Transfer Gap ($\mathrm{F1_{ctrl}} - \mathrm{F1_{wild}}$)")
        ax.set_title(get_display_name(exp_name))

    # Add legend to first subplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GEOMETRIC_COLOR, label="Geometric"),
        Patch(facecolor=HOLISTIC_COLOR, label="Holistic"),
        Patch(facecolor=TEMPORAL_COLOR, label="Temporal"),
    ]
    axes[0][0].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_transfer_gap.pdf")
    fig.savefig(out_dir / "fig4_transfer_gap.png")
    plt.close(fig)
    print("  Generated fig4_transfer_gap.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Calibration scatter (E5 vs E5b)
# ═══════════════════════════════════════════════════════════════════════
def generate_calibration_scatter(all_metrics: dict, out_dir: Path):
    """Fig 5: Calibration scatter comparing agentic experiments."""
    agentic_exps = {}
    for (exp, domain), m in all_metrics.items():
        if "agentic" in exp.lower() and "calibration" in m and domain == "controlled":
            agentic_exps[exp] = m

    if len(agentic_exps) < 1:
        print("  Skipping fig5 (no agentic experiments with calibration)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Tool invocation rate by field
    ax1 = axes[0]
    exp_names = sorted(agentic_exps.keys())
    fields = None
    bar_data = {}

    for exp in exp_names:
        cal = agentic_exps[exp].get("calibration", {})
        if fields is None:
            fields = sorted(cal.keys())
        bar_data[exp] = [cal.get(f, {}).get("tool_invocation_rate", 0) * 100 for f in fields]

    x = np.arange(len(fields))
    width = 0.8 / len(exp_names)
    for i, exp in enumerate(exp_names):
        offset = (i - len(exp_names) / 2 + 0.5) * width
        ax1.bar(x + offset, bar_data[exp], width, label=get_display_name(exp), color=get_color(exp), edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(fields, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Tool Invocation Rate (%)")
    ax1.set_title("(a) Tool Invocation Rate by Field")
    ax1.legend(fontsize=8)

    # Plot 2: Missed opportunity rate vs accuracy when not invoked
    ax2 = axes[1]
    for exp in exp_names:
        cal = agentic_exps[exp].get("calibration", {})
        for field, data in cal.items():
            acc = data.get("accuracy_when_not_invoked", 0) * 100
            missed = data.get("missed_opportunity_rate", 0) * 100
            color = get_color(exp)
            if field in ("stance", "depth"):
                marker = "o"
            else:
                marker = "s"
            ax2.scatter(acc, missed, c=color, marker=marker, s=80, edgecolors="black", linewidth=0.5, zorder=3)
            ax2.annotate(field, (acc, missed), fontsize=7, ha="left", va="bottom",
                         xytext=(3, 3), textcoords="offset points")

    # Add manual legend
    from matplotlib.lines import Line2D
    legend1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=get_color(e), markersize=8,
                       markeredgecolor="black", label=get_display_name(e)) for e in exp_names]
    legend2 = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Geometric"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="Holistic/Other")]
    ax2.legend(handles=legend1 + legend2, fontsize=7, loc="upper right")

    ax2.set_xlabel("Accuracy When Tool Not Invoked (%)")
    ax2.set_ylabel("Missed Opportunity Rate (%)")
    ax2.set_title("(b) Calibration: Accuracy vs Missed Opportunities")
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_calibration.pdf")
    fig.savefig(out_dir / "fig5_calibration.png")
    plt.close(fig)
    print("  Generated fig5_calibration.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: F1 heatmap
# ═══════════════════════════════════════════════════════════════════════
def generate_f1_heatmap(all_metrics: dict, out_dir: Path):
    """Fig 6: F1 heatmap (experiments x labels)."""
    controlled = {exp: m for (exp, domain), m in all_metrics.items() if domain == "controlled"}
    if not controlled:
        return

    exp_names = sorted(controlled.keys())
    matrix = np.zeros((len(exp_names), len(ALL_LABELS)))

    for i, exp in enumerate(exp_names):
        for j, label in enumerate(ALL_LABELS):
            matrix[i, j] = controlled[exp]["per_label"].get(label, {}).get("f1", 0)

    fig, ax = plt.subplots(figsize=(14, max(4, len(exp_names) * 0.9)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ALL_LABELS)))
    ax.set_xticklabels(ALL_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels([get_display_name(e) for e in exp_names], fontsize=9)

    for i in range(len(exp_names)):
        for j in range(len(ALL_LABELS)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("F1 Score")

    # Group separators
    geo_end = len([l for l in ALL_LABELS if l in GEOMETRIC_LABELS]) - 0.5
    hol_end = geo_end + len([l for l in ALL_LABELS if l in HOLISTIC_LABELS])
    ax.axvline(x=geo_end, color="white", linewidth=2)
    ax.axvline(x=hol_end, color="white", linewidth=2)

    # Group labels at top
    geo_mid = geo_end / 2
    hol_mid = (geo_end + hol_end) / 2
    other_mid = (hol_end + len(ALL_LABELS) - 1) / 2
    ax.text(geo_mid, -0.7, "Geometric", ha="center", fontsize=9, fontweight="bold", color=GEOMETRIC_COLOR)
    ax.text(hol_mid, -0.7, "Holistic", ha="center", fontsize=9, fontweight="bold", color=HOLISTIC_COLOR)
    ax.text(other_mid, -0.7, "Other", ha="center", fontsize=9, fontweight="bold", color=TEMPORAL_COLOR)

    ax.set_title("Per-Label F1 Across Experiments (Controlled Domain)", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_f1_heatmap.pdf")
    fig.savefig(out_dir / "fig6_f1_heatmap.png")
    plt.close(fig)
    print("  Generated fig6_f1_heatmap.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Radar chart
# ═══════════════════════════════════════════════════════════════════════
def generate_radar_chart(all_metrics: dict, out_dir: Path):
    """Fig 7: Radar chart comparing models across key metrics."""
    controlled = {exp: m for (exp, domain), m in all_metrics.items() if domain == "controlled"}
    if not controlled:
        return

    # Metrics for radar (all on 0-1 scale)
    metric_names = ["Macro F1", "Micro F1", "Subset Acc", "Stance Acc", "Depth Acc", "Geo F1", "Hol F1", "Coverage"]
    metric_keys = [
        lambda m: m["macro"]["f1"],
        lambda m: m["micro"]["f1"],
        lambda m: m.get("subset_accuracy", 0),
        lambda m: m.get("stance_accuracy", 0),
        lambda m: m.get("depth_accuracy", 0),
        lambda m: m.get("geometric_avg_f1", 0),
        lambda m: m.get("holistic_avg_f1", 0),
        lambda m: m.get("label_coverage", 0),
    ]

    exp_names = sorted(controlled.keys())
    n_metrics = len(metric_names)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for exp in exp_names:
        m = controlled[exp]
        values = [fn(m) for fn in metric_keys]
        values += values[:1]  # Close
        color = get_color(exp)
        ax.plot(angles, values, "o-", linewidth=1.5, label=get_display_name(exp), color=color, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, alpha=0.7)
    ax.set_title("Model Comparison (Controlled Domain)", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig7_radar.pdf")
    fig.savefig(out_dir / "fig7_radar.png")
    plt.close(fig)
    print("  Generated fig7_radar.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# BONUS: Main results bar chart with error bars
# ═══════════════════════════════════════════════════════════════════════
def generate_main_results_bar_chart(all_metrics: dict, out_dir: Path):
    """Fig 1: Main results bar chart — Macro F1 across experiments with CI error bars."""
    # Group by domain
    for domain in ["controlled", "wild"]:
        domain_data = {exp: m for (exp, d), m in all_metrics.items() if d == domain}
        if not domain_data:
            continue

        exp_names = sorted(domain_data.keys())
        macro_f1s = [domain_data[e].get("macro_f1_active", domain_data[e]["macro"]["f1"]) for e in exp_names]
        ci_lows = []
        ci_highs = []
        for e in exp_names:
            ci = domain_data[e].get("bootstrap_ci", {}).get("macro_f1_active",
                 domain_data[e].get("bootstrap_ci", {}).get("macro_f1", {}))
            f1 = domain_data[e].get("macro_f1_active", domain_data[e]["macro"]["f1"])
            ci_lows.append(f1 - ci.get("ci_low", f1))
            ci_highs.append(ci.get("ci_high", f1) - f1)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(exp_names))
        colors = [get_color(e) for e in exp_names]

        bars = ax.bar(x, macro_f1s, color=colors, edgecolor="black", linewidth=0.5,
                      yerr=[ci_lows, ci_highs], capsize=4, error_kw={"linewidth": 1.2})

        # Value labels
        for bar, val in zip(bars, macro_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_highs or [0]) + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([get_display_name(e) for e in exp_names], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Macro F1 (active labels)")
        ax.set_ylim(0, min(1.0, max(macro_f1s) * 1.4 + 0.05))
        ax.set_title(f"Macro F1 (Active Labels) — {domain.capitalize()} Domain")
        ax.axhline(y=0, color="gray", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(out_dir / f"fig1_macro_f1_{domain}.pdf")
        fig.savefig(out_dir / f"fig1_macro_f1_{domain}.png")
        plt.close(fig)
        print(f"  Generated fig1_macro_f1_{domain}.pdf/png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.experiments_dir)

    # Collect all metrics
    all_metrics = {}
    if exp_dir.exists():
        for subdir in sorted(exp_dir.iterdir()):
            if not subdir.is_dir() or subdir.name in ("statistical_summary", "transfer_analysis"):
                continue
            for domain in ["controlled", "wild"]:
                m = load_metrics(exp_dir, subdir.name, domain)
                if m:
                    all_metrics[(subdir.name, domain)] = m
            # Also try direct (no domain subdir)
            direct = subdir / "metrics.json"
            if direct.exists() and (subdir.name, "controlled") not in all_metrics:
                with open(direct) as f:
                    m = json.load(f)
                domain = m.get("domain", "controlled")
                all_metrics[(subdir.name, domain)] = m

    if not all_metrics:
        print("No experiment results found. Tables and figures will be generated once experiments are run.")
        return

    print(f"Found {len(all_metrics)} result sets:")
    for (exp, domain) in sorted(all_metrics.keys()):
        macro_f1 = all_metrics[(exp, domain)]["macro"]["f1"]
        print(f"  {exp}/{domain}: Macro F1 = {macro_f1:.3f}")

    print("\nGenerating tables...")
    generate_main_results_table(all_metrics, out_dir)
    generate_per_label_table(all_metrics, out_dir)
    generate_transfer_gap_table(all_metrics, out_dir)
    generate_calibration_table(all_metrics, out_dir)
    generate_coaching_table(all_metrics, exp_dir, out_dir)
    generate_mcnemar_table(exp_dir, out_dir)

    print("\nGenerating figures...")
    generate_main_results_bar_chart(all_metrics, out_dir)
    generate_label_distribution(Path(args.train_data), out_dir)
    generate_per_label_bar_chart(all_metrics, out_dir)
    generate_transfer_gap_figure(all_metrics, out_dir)
    generate_calibration_scatter(all_metrics, out_dir)
    generate_f1_heatmap(all_metrics, out_dir)
    generate_radar_chart(all_metrics, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
    print(f"  Tables: table2-7 (.md + .tex)")
    print(f"  Figures: fig1-7 (.pdf + .png)")


if __name__ == "__main__":
    main()
