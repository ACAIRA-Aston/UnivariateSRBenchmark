"""
Generate aggregated extrapolation box plots across tiers
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Path configurations
RUNBENCH_DIR = Path(__file__).parent
BENCHMARK_RESULTS_PATH = RUNBENCH_DIR / "benchmark_results"
OUTPUT_PLOTS_DIR = RUNBENCH_DIR / "plots"

TIERS = ["Easy", "Medium", "Hard"]
SUBPLOT_LABELS = ["(a)", "(b)", "(c)"]

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

DEFAULT_MSE_MAX_THRESHOLDS: Union[float, None, Dict[str, Optional[float]]] = {
    "Easy": 1e4,
    "Medium": 1e4,
    "Hard": 1e4,
}

DEFAULT_R2_MIN_THRESHOLDS: Union[float, None, Dict[str, Optional[float]]] = {
    "Easy": None,
    "Medium": -10.0,
    "Hard": -1e4,
}

R2_UPPER_BOUND = 0.9999


def resolve_tier_threshold(
    threshold_param: Union[float, None, Dict[str, Optional[float]]],
    tier: str,
    default_val: Optional[float] = None
) -> Optional[float]:
    """Extracts threshold for a given tier"""
    if threshold_param is None:
        return None
    if isinstance(threshold_param, dict):
        val = threshold_param.get(tier, default_val)
    elif isinstance(threshold_param, (int, float)):
        val = float(threshold_param)
    else:
        val = default_val

    if val is None or math.isnan(val) or math.isinf(val):
        return default_val
    return val


def get_gradient_colors(n: int, reverse: bool = False) -> list:
    cmap = plt.cm.get_cmap("Reds")
    vals = np.linspace(0.35, 0.90, n)
    if reverse:
        vals = vals[::-1]
    return [cmap(v) for v in vals]


def load_benchmark_data() -> pd.DataFrame:
    """Reads JSON result files and extracts extrapolation MSE and R² values per problem."""
    records = []
    for tier in TIERS:
        tier_dir = BENCHMARK_RESULTS_PATH / tier
        if not tier_dir.exists():
            continue
        for method in METHODS:
            method_dir = tier_dir / method
            if not method_dir.exists():
                continue
            for json_file in method_dir.glob("*.json"):
                problem_id = json_file.stem
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    metrics = data.get("summary_metrics", {})

                    e_mse = metrics.get("extra_mse_mean")
                    e_r2 = metrics.get("extra_r2_mean")

                    if e_mse is None or not isinstance(e_mse, (int, float)) or math.isnan(e_mse) or math.isinf(e_mse):
                        e_mse = np.nan
                    if e_r2 is None or not isinstance(e_r2, (int, float)) or math.isnan(e_r2) or math.isinf(e_r2):
                        e_r2 = np.nan

                    records.append({
                        "tier": tier,
                        "algorithm": method,
                        "problem_id": problem_id,
                        "extra_mse": e_mse,
                        "extra_r2": e_r2
                    })
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

    return pd.DataFrame(records)


def transform_r2(r2):
    """Transform R² using log1p(-r2) / log(10)."""
    return -np.log1p(-r2) / np.log(10.0)


def format_r2_label(val: float, is_min_thresh: bool = False) -> str:
    """Formats R² tick labels nicely using LaTeX notation for exponents."""
    prefix = "<" if is_min_thresh else ""
    if val == 0:
        return f"{prefix}0"

    abs_val = abs(val)
    sign = "-" if val < 0 else ""

    log10_val = np.log10(abs_val)
    if math.isclose(log10_val, round(log10_val), abs_tol=1e-4):
        k = int(round(log10_val))
        if k == 0:
            return f"{prefix}{sign}1"
        elif k == 1 and val > 0:
            return f"{prefix}10"
        else:
            return rf"${prefix}{sign}10^{{{k}}}$"

    if 1e-3 <= abs_val <= 1.0:
        return f"{prefix}{val:g}"

    exponent = int(np.floor(np.log10(abs_val)))
    mantissa = abs_val / (10 ** exponent)
    if math.isclose(mantissa, 1.0, rel_tol=1e-2):
        return rf"${prefix}{sign}10^{{{exponent}}}$"
    return rf"${prefix}{sign}{mantissa:g} \times 10^{{{exponent}}}$"


def generate_tier_r2_ticks(min_thresh: Optional[float], actual_min: float):
    """Generates reduced R² tick marks to fit comfortably in a 10x4 figure."""
    # Simplified set of positive ticks to prevent clutter
    positive_ticks = [0.8, 0.99, 0.999, 0.9999]
    start_val = min_thresh if min_thresh is not None else actual_min

    ticks = []
    if min_thresh is not None:
        ticks.append(min_thresh)

    if start_val < 0:
        # Step negative exponents by 2 or 3 powers of 10 to reduce tick density
        if start_val <= -10:
            min_exp = int(np.floor(np.log10(abs(start_val))))
            step = max(2, min_exp // 3)
            for exp in range(min_exp - step, 0, -step):
                val = -(10.0 ** exp)
                if val > start_val:
                    ticks.append(val)
        ticks.append(0.0)
    elif start_val == 0:
        ticks.append(0.0)

    for t in positive_ticks:
        if t > start_val:
            ticks.append(t)

    unique_ticks = []
    for t in ticks:
        if not any(math.isclose(t, ut, abs_tol=1e-12, rel_tol=1e-4) for ut in unique_ticks):
            unique_ticks.append(t)

    labels = []
    for i, t in enumerate(unique_ticks):
        is_clipped_label = (i == 0 and min_thresh is not None)
        labels.append(format_r2_label(t, is_min_thresh=is_clipped_label))

    return unique_ticks, labels


# ==============================================================================
# 1. R² Distribution Boxplot
# ==============================================================================
def plot_r2_extrapolation_all_tiers(
    df: pd.DataFrame,
    r2_min_thresholds: Union[float, None, Dict[str, Optional[float]]] = DEFAULT_R2_MIN_THRESHOLDS,
    save_path: Path = None, dark_mode: bool = False
):
    """Plot R² extrapolation boxplots formatted for compact figures."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.patch.set_alpha(0.0)  # Set figure background transparent
    for ax in axes:
        ax.patch.set_alpha(0.0)

    for idx, tier in enumerate(TIERS):
        ax = axes[idx]
        tier_df = df[(df['tier'] == tier) & (df['extra_r2'].notna())].copy()

        tier_r2_min = resolve_tier_threshold(r2_min_thresholds, tier, default_val=None)

        if tier_df.empty:
            continue

        actual_min_r2 = tier_df['extra_r2'].min()

        algo_order = (
            tier_df.groupby('algorithm')['extra_r2']
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )

        colors = get_gradient_colors(len(algo_order), reverse=False)

        for i, algo in enumerate(algo_order):
            algo_data = tier_df[tier_df['algorithm'] == algo]['extra_r2'].values
            if len(algo_data) == 0:
                continue

            if tier_r2_min is not None:
                algo_data_processed = np.clip(algo_data, tier_r2_min, R2_UPPER_BOUND)
            else:
                algo_data_processed = np.minimum(algo_data, R2_UPPER_BOUND)

            transformed_data = transform_r2(algo_data_processed)

            ax.boxplot(
                transformed_data,
                positions=[i],
                vert=False,
                widths=0.55,
                patch_artist=True,
                showmeans=False,
                showfliers=True,
                flierprops=dict(marker='o', markersize=2.5, markerfacecolor=colors[i], markeredgecolor='none', alpha=0.35),
                boxprops=dict(facecolor=colors[i], color='darkred', alpha=0.8, linewidth=1.0),
                whiskerprops=dict(color='darkred', linewidth=1.0),
                capprops=dict(color='darkred', linewidth=1.0),
                medianprops=dict(color='black', linewidth=1.4)
            )

        r2_ticks, tick_labels = generate_tier_r2_ticks(tier_r2_min, actual_min_r2)
        transformed_ticks = [transform_r2(r) for r in r2_ticks]

        ax.set_xticks(transformed_ticks)
        ax.set_xticklabels(tick_labels)

        # Rotate tick labels diagonally to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')

        ax.set_yticks(np.arange(len(algo_order)))
        ax.set_yticklabels(algo_order, fontsize=10)

        ax.set_xlabel(f"{SUBPLOT_LABELS[idx]}", fontsize=10.5, labelpad=4)

        effective_min = tier_r2_min if tier_r2_min is not None else actual_min_r2
        min_x_limit = transform_r2(effective_min)
        max_x_limit = transform_r2(R2_UPPER_BOUND)
        ax.set_xlim(min_x_limit - 0.15, max_x_limit + 0.15)

        ax.xaxis.grid(True, alpha=0.3, linewidth=0.8)
        ax.yaxis.grid(False)

        for r2_val in [0.0, 0.9, 0.99]:
            if r2_val > effective_min:
                ax.axvline(x=transform_r2(r2_val), color='gray', linestyle='--', alpha=0.25, linewidth=0.8)

        sns.despine(ax=ax, left=True, bottom=True)
    if dark_mode:
        apply_dark_theme(fig, axes)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# ==============================================================================
# 2. MSE Distribution Boxplot
# ==============================================================================
def plot_mse_extrapolation_all_tiers(
    df: pd.DataFrame,
    mse_max_thresholds: Union[float, None, Dict[str, Optional[float]]] = DEFAULT_MSE_MAX_THRESHOLDS,
    save_path: Path = None, dark_mode: bool = False
):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)

    for idx, tier in enumerate(TIERS):
        ax = axes[idx]
        tier_df = df[(df['tier'] == tier) & (df['extra_mse'].notna()) & (df['extra_mse'] >= 0)].copy()

        tier_mse_max = resolve_tier_threshold(mse_max_thresholds, tier, default_val=None)

        if tier_df.empty:
            continue

        actual_max_mse = tier_df['extra_mse'].max()

        algo_order = (
            tier_df.groupby('algorithm')['extra_mse']
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        colors = get_gradient_colors(len(algo_order), reverse=True)

        for i, algo in enumerate(algo_order):
            algo_data = tier_df[tier_df['algorithm'] == algo]['extra_mse'].values
            if len(algo_data) == 0:
                continue

            if tier_mse_max is not None:
                algo_data_processed = np.minimum(algo_data, tier_mse_max)
            else:
                algo_data_processed = algo_data

            ax.boxplot(
                algo_data_processed,
                positions=[i],
                vert=False,
                widths=0.55,
                patch_artist=True,
                showmeans=False,
                showfliers=True,
                flierprops=dict(marker='o', markersize=2.5, markerfacecolor=colors[i], markeredgecolor='none', alpha=0.35),
                boxprops=dict(facecolor=colors[i], color='darkred', alpha=0.8, linewidth=1.0),
                whiskerprops=dict(color='darkred', linewidth=1.0),
                capprops=dict(color='darkred', linewidth=1.0),
                medianprops=dict(color='black', linewidth=1.4)
            )

        lin_threshold = 1e-3
        ax.set_xscale('symlog', linthresh=lin_threshold)

        def create_symlog_formatter(max_thresh):
            def formatter(x, _):
                if x < 0:
                    return ''
                if math.isclose(x, 0.0, abs_tol=1e-12):
                    return '$0$'
                if max_thresh is not None and x >= max_thresh - 1e-3:
                    log_val = int(round(np.log10(max_thresh)))
                    return rf'$>10^{{{log_val}}}$' if math.isclose(max_thresh, 10**log_val) else f'$>{max_thresh:g}$'
                try:
                    log_val = int(round(np.log10(x)))
                    if math.isclose(x, 10 ** log_val, rel_tol=1e-2):
                        return rf'$10^{{{log_val}}}$'
                    return ''
                except Exception:
                    return ''
            return formatter

        effective_max = tier_mse_max if tier_mse_max is not None else actual_max_mse
        log_max = int(np.ceil(np.log10(max(effective_max, 1.0))))

        major_ticks = [0.0] + [10.0 ** k for k in range(-3, log_max + 1, 2) if (tier_mse_max is None or 10.0 ** k <= tier_mse_max)]
        if tier_mse_max is not None and not any(math.isclose(tier_mse_max, t) for t in major_ticks):
            major_ticks.append(tier_mse_max)

        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(create_symlog_formatter(tier_mse_max)))

        plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')

        right_lim = (tier_mse_max * 1.3) if tier_mse_max is not None else (actual_max_mse * 1.3)
        ax.set_xlim(left=0, right=right_lim)

        ax.set_yticks(np.arange(len(algo_order)))
        ax.set_yticklabels(algo_order, fontsize=10)
        ax.set_xlabel(f"{SUBPLOT_LABELS[idx]}", fontsize=10.5, labelpad=4)

        ax.xaxis.grid(True, alpha=0.3, linewidth=0.8, which='both')
        ax.yaxis.grid(False)

        sns.despine(ax=ax, left=True, bottom=True)
    if dark_mode:
        apply_dark_theme(fig, axes)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def apply_dark_theme(fig, axes):
    """Adjusts text, axes, ticks, and grids for dark backgrounds."""
    fig.patch.set_facecolor('none')
    for ax in axes:
        ax.patch.set_facecolor('none')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        # Make tick labels light
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')

        # Subtle light grid lines for visibility on dark gray/black
        ax.xaxis.grid(True, color='white', alpha=0.15, linewidth=0.8)

        # Spines/borders color
        for spine in ax.spines.values():
            spine.set_color('white')


def main():
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark results from JSON files...")
    df = load_benchmark_data()

    if df.empty:
        print("Error: No benchmark results found.")
        return

    custom_r2_thresholds = {"Easy": -1e7, "Medium": -1e8, "Hard": -1e10}
    custom_mse_thresholds = {"Easy": 1e4, "Medium": 1e4, "Hard": 1e4}

    # ------------------------------------------------------------------
    # Light Mode
    # ------------------------------------------------------------------
    print("Generating Figure: Extrapolation R² Boxplots (Light / Transparent)...")
    r2_plot_path = OUTPUT_PLOTS_DIR / "extrapolation_r2_tiers.png"
    plot_r2_extrapolation_all_tiers(
        df,
        r2_min_thresholds=custom_r2_thresholds,
        save_path=r2_plot_path,
        dark_mode=False
    )
    print(f"✓ Saved: {r2_plot_path}")

    print("Generating Figure: Extrapolation MSE Boxplots (Light / Transparent)...")
    mse_plot_path = OUTPUT_PLOTS_DIR / "extrapolation_mse_tiers.png"
    plot_mse_extrapolation_all_tiers(
        df,
        mse_max_thresholds=custom_mse_thresholds,
        save_path=mse_plot_path,
        dark_mode=False
    )
    print(f"✓ Saved: {mse_plot_path}")

    # ------------------------------------------------------------------
    # Dark Mode
    # ------------------------------------------------------------------
    print("Generating Figure: Extrapolation R² Boxplots (Dark / Transparent)...")
    r2_dark_path = OUTPUT_PLOTS_DIR / "extrapolation_r2_tiers_dark.png"
    plot_r2_extrapolation_all_tiers(
        df,
        r2_min_thresholds=custom_r2_thresholds,
        save_path=r2_dark_path,
        dark_mode=True
    )
    print(f"✓ Saved: {r2_dark_path}")

    print("Generating Figure: Extrapolation MSE Boxplots (Dark / Transparent)...")
    mse_dark_path = OUTPUT_PLOTS_DIR / "extrapolation_mse_tiers_dark.png"
    plot_mse_extrapolation_all_tiers(
        df,
        mse_max_thresholds=custom_mse_thresholds,
        save_path=mse_dark_path,
        dark_mode=True
    )
    print(f"✓ Saved: {mse_dark_path}")


if __name__ == "__main__":
    main()