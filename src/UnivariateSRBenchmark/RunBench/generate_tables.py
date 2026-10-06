"""
Generate LaTeX tables for Symbolic Regression Benchmark results.
Location: UnivariateSRBenchmark/RunBench/generate_latex_tables.py
"""

import json
import math
from pathlib import Path
import pandas as pd

# Path configurations
RUNBENCH_DIR = Path(__file__).parent
BENCHMARK_RESULTS_PATH = RUNBENCH_DIR / "benchmark_results"
OUTPUT_TEX_DIR = RUNBENCH_DIR / "latex_tables"

TIERS = ["Easy", "Medium", "Hard"]

TIER_PREFIX_MAP = {
    "Easy": "E",
    "Medium": "M",
    "Hard": "H",
}

METHODS = [
    "AIFeynman",
    "BSR",
    "DSR",
    "E2E",
    "GPGOMEA",
    "Gplearn",
    "ITEA",
    "NeSymReS",
    "Operon",
    "PySR",
    "TaylorGP",
    "TPSR",
    "UDSR",
]

INTER_DATASET_FILES = {
    "Easy": Path("../Datasets/interpolation_easy.parquet"),
    "Medium": Path("../Datasets/interpolation_medium.parquet"),
    "Hard": Path("../Datasets/interpolation_hard.parquet"),
}


def transform_problem_id(pid: str) -> str:
    """Transforms problem ID format from 'Easy-40' to 'E41' (1-based index)."""
    try:
        tier_name, idx_str = pid.split("-")
        prefix = TIER_PREFIX_MAP.get(tier_name, tier_name[0])
        idx = int(idx_str) + 1
        return f"{prefix}{idx}"
    except Exception:
        return pid


def format_val(v: float) -> str:
    """Formats a single float value according to scale-based precision rules."""
    if v is None or math.isnan(v) or math.isinf(v):
        return "---"
    if abs(v) < 1e-12:
        return "0"

    abs_v = abs(v)

    # Exponential scientific notation for extreme magnitudes (< 0.01 or >= 10,000)
    if abs_v < 1e-2 or abs_v >= 1e4:
        formatted = f"{v:.1e}"
        formatted = formatted.replace("e-0", "e-").replace("e+0", "e")
        return formatted

    # Scale-based decimal formatting:
    if abs_v >= 100:
        return f"{v:.1f}"
    elif abs_v >= 1:
        return f"{v:.2f}"
    else:
        return f"{v:.3f}"


def format_mean_std(mean: float, std: float) -> str:
    """Formats mean and std as 'mean $\\pm$ std' or '---' if either is NaN/invalid."""
    if mean is None or std is None:
        return "---"
    if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
        return "---"
    if math.isnan(mean) or math.isnan(std) or math.isinf(mean) or math.isinf(std):
        return "---"

    m_str = format_val(mean)
    s_str = format_val(std)

    if m_str == "---" or s_str == "---":
        return "---"

    return f"{m_str} $\\pm$ {s_str}"


def get_problem_ids(tier: str) -> list[str]:
    """Retrieves all original problem IDs for a tier from dataset parquet or existing JSONs."""
    parquet_path = (RUNBENCH_DIR / INTER_DATASET_FILES[tier]).resolve()
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return [f"{tier}-{i}" for i in range(len(df))]

    # Fallback to scanning existing benchmark JSON files
    tier_dir = BENCHMARK_RESULTS_PATH / tier
    problem_ids = set()
    if tier_dir.exists():
        for method in METHODS:
            method_dir = tier_dir / method
            if method_dir.exists():
                for f in method_dir.glob("*.json"):
                    problem_ids.add(f.stem)

    def sort_key(pid: str):
        try:
            return int(pid.split("-")[-1])
        except ValueError:
            return pid

    return sorted(list(problem_ids), key=sort_key)


def generate_latex_table(tier: str, metric_type: str, problem_ids: list[str]) -> str:
    """Generates a LaTeX table string for a specific tier and metric ('MSE' or 'R2')."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    if metric_type == "R2":
        metric_type_tex = "$R^2$"
    else:
        metric_type_tex = metric_type
    lines.append(f"\\caption{{{tier} Tier — {metric_type_tex} Performance}}")
    lines.append(f"\\label{{tab:{tier.lower()}_{metric_type.lower()}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    col_spec = "@{}cc" + "c" * len(METHODS) + "@{}"

    lines.append(f"\\begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append(r"\toprule")

    headers = ["ID", "Set"] + METHODS
    lines.append(" & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\")
    lines.append(r"\midrule")

    metric_key = metric_type.lower()
    mean_key_inter = f"inter_{metric_key}_mean"
    std_key_inter = f"inter_{metric_key}_std"
    mean_key_extra = f"extra_{metric_key}_mean"
    std_key_extra = f"extra_{metric_key}_std"

    for idx, pid in enumerate(problem_ids):
        formatted_pid = transform_problem_id(pid)
        inter_vals = []
        extra_vals = []

        for method in METHODS:
            json_file = BENCHMARK_RESULTS_PATH / tier / method / f"{pid}.json"

            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    metrics = data.get("summary_metrics", {})

                    i_m = metrics.get(mean_key_inter)
                    i_s = metrics.get(std_key_inter)
                    e_m = metrics.get(mean_key_extra)
                    e_s = metrics.get(std_key_extra)

                    inter_vals.append(format_mean_std(i_m, i_s))
                    extra_vals.append(format_mean_std(e_m, e_s))
                except Exception:
                    inter_vals.append("")
                    extra_vals.append("")
            else:
                inter_vals.append("")
                extra_vals.append("")

        inter_row = f"\\multirow{{2}}{{*}}{{{formatted_pid}}} & Int. & " + " & ".join(inter_vals) + r" \\"
        extra_row = f" & Ext. & " + " & ".join(extra_vals) + r" \\"

        lines.append(inter_row)
        lines.append(extra_row)

        if idx < len(problem_ids) - 1:
            lines.append(r"\cline{2-" + str(2 + len(METHODS)) + "}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    OUTPUT_TEX_DIR.mkdir(parents=True, exist_ok=True)

    for tier in TIERS:
        problem_ids = get_problem_ids(tier)
        if not problem_ids:
            print(f"Warning: No problem IDs found for tier {tier}.")
            continue

        for metric in ["MSE", "R2"]:
            latex_code = generate_latex_table(tier, metric, problem_ids)
            out_filename = OUTPUT_TEX_DIR / f"{tier.lower()}_{metric.lower()}_table.tex"

            with open(out_filename, "w", encoding="utf-8") as f:
                f.write(latex_code)

            print(f"✓ Saved: {out_filename}")


if __name__ == "__main__":
    main()