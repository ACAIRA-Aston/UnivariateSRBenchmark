"""
Benchmark execution runner using optimal hyperparameter configurations.
"""
import os

# Add MSYS2 core utilities (head.exe, cat.exe, rm.exe) to PATH if on Windows
msys_usr_bin = r"C:\msys64\usr\bin"
git_usr_bin = r"C:\Program Files\Git\usr\bin"
if os.path.exists(msys_usr_bin):
    os.environ["PATH"] = msys_usr_bin + os.pathsep + os.environ["PATH"]
elif os.path.exists(git_usr_bin):
    os.environ["PATH"] = git_usr_bin + os.pathsep + os.environ["PATH"]

import sys
import json
import importlib
import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from typing import Type, Any, Optional
from SRToolkit.dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary
from UnivariateSRBenchmark.utils import sympy_to_srtoolkit_tokens, get_project_root
from UnivariateSRBenchmark.config import *
# Directories and Paths
RUNBENCH_DIR = Path(__file__).parent
TUNING_GRID_RESULTS_PATH = RUNBENCH_DIR.parent / "Tuning" / "grid_results"
BENCHMARK_RESULTS_PATH = RUNBENCH_DIR / "benchmark_results"


def get_approach_class(name: str) -> Type:
    """Dynamically imports and returns the approach class for the given method name."""
    if name not in APPROACH_REGISTRY:
        valid_methods = ", ".join(APPROACH_REGISTRY.keys())
        raise KeyError(f"Unknown approach '{name}'. Valid options: {valid_methods}")

    module_path, class_name = APPROACH_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_best_config(method: str, tier: str) -> Dict[str, Any]:
    """Loads the optimal hyperparameter configuration saved in best_run.json."""
    tier_cap = tier.capitalize()
    default_problem = DEFAULT_TUNING_PROBLEMS.get(tier_cap, "Easy-9")
    best_run_path = TUNING_GRID_RESULTS_PATH / default_problem / method / "best_run.json"

    if not best_run_path.exists():
        raise FileNotFoundError(
            f"Could not find best_run.json for {method} at '{best_run_path}'. "
            f"Ensure grid search tuning and summary rebuilding have been completed.")
    with open(best_run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["config"]


def evaluate_expression(expr_input: Any, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Evaluates a learned SymPy expression or string on X and computes MSE and R^2 against y.
    Returns (MSE, R^2).
    """
    if expr_input is None or str(expr_input) in ("None", "nan", "", "NoneType"):
        return np.nan, np.nan

    try:
        expr = sp.sympify(str(expr_input))
        n_features = X.shape[1] if X.ndim > 1 else 1

        # Map features to symbols X_0, X_1, ...
        symbols = [sp.Symbol(f"X_{i}") for i in range(n_features)]
        f = sp.lambdify(symbols, expr,
                        modules=["numpy", {"square": np.square, "cube": lambda x: x ** 3, "powu": np.power}])
        if n_features == 1:
            y_pred = f(X[:, 0])
        else:
            y_pred = f(*[X[:, i] for i in range(n_features)])

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 0 or y_pred.shape != y.shape:
            y_pred = np.full_like(y, y_pred)
        if not np.all(np.isfinite(y_pred)):
            return np.nan, np.nan

        mse = float(np.mean((y - y_pred) ** 2))

        # Calculate R^2 score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

        return mse, r2

    except Exception:
        return np.nan, np.nan


def load_problem_data(tier: str, row_index: int) -> Tuple[SR_dataset, np.ndarray, np.ndarray, str]:
    """
    Loads interpolation and extrapolation data for a specific problem instance.
    Returns: (interpolation_dataset, X_extra, y_extra, original_equation)
    """
    tier_cap = tier.capitalize()
    inter_path = get_project_root() / INTER_DATASET_FILES[tier_cap]
    extra_path = get_project_root() / EXTRA_DATASET_FILES[tier_cap]

    if not inter_path.exists() or not extra_path.exists():
        raise FileNotFoundError(f"Dataset parquet files not found for tier {tier_cap}.")

    df_inter = pd.read_parquet(inter_path)
    df_extra = pd.read_parquet(extra_path)

    if row_index < 0 or row_index >= len(df_inter):
        raise IndexError(f"Row index {row_index} out of bounds for dataset tier '{tier_cap}' (size {len(df_inter)}).")

    row_inter = df_inter.iloc[row_index]
    row_extra = df_extra.iloc[row_index]

    equation_str = row_inter["equation"]

    # Interpolation Data
    X_inter = np.asarray(row_inter["X"], dtype=float)
    y_inter = np.asarray(row_inter["Y"], dtype=float)
    if X_inter.ndim == 1:
        X_inter = X_inter.reshape(-1, 1)

    # Extrapolation Data
    X_extra = np.asarray(row_extra["X"], dtype=float)
    y_extra = np.asarray(row_extra["Y"], dtype=float)
    if X_extra.ndim == 1:
        X_extra = X_extra.reshape(-1, 1)

    n_features = X_inter.shape[1]
    sl = SymbolLibrary.default_symbols(num_variables=n_features)

    # Ground truth token conversion
    ground_truth_tokens = None
    try:
        X_syms = {f"X{i}": sp.Symbol(f"X{i}", real=True) for i in range(n_features)}
        expr = sp.sympify(equation_str, locals=X_syms)
        expr = expr.subs({sp.Symbol(f"X{i}"): sp.Symbol(f"X_{i}") for i in range(n_features)})
        ground_truth_tokens, _ = sympy_to_srtoolkit_tokens(expr)
    except Exception:
        pass

    problem_id = f"{tier_cap}-{row_index}"

    inter_dataset = SR_dataset(
        X=X_inter,
        y=y_inter,
        symbol_library=sl,
        dataset_name=problem_id,
        original_equation=str(equation_str),
        ground_truth=ground_truth_tokens,
        max_evaluations=50000,
        success_threshold=1e-6,
        constant_bounds=(-100.0, 100.0),
        max_expr_length=150,
    )

    return inter_dataset, X_extra, y_extra, equation_str


def run_benchmark(
        method: str,
        tier: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        num_experiments: int = 10,
) -> None:
    """
    Executes benchmark runs using optimal hyperparameter configuration across dataset problems.
    """
    tier_cap = tier.capitalize()

    # Load optimal config and approach class
    best_config = load_best_config(method, tier_cap)
    approach_cls = get_approach_class(method)

    # Output directory: benchmark_results/<Tier>/<Method>/
    out_dir = BENCHMARK_RESULTS_PATH / tier_cap / method
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine problem count from dataset
    inter_path = get_project_root() / INTER_DATASET_FILES[tier_cap]
    total_problems = len(pd.read_parquet(inter_path))

    start_i = max(0, start_idx) if start_idx is not None else 0
    end_i = min(total_problems, end_idx + 1) if end_idx is not None else total_problems

    # Deterministic execution override
    if method in {"AIFeynman", "DSR"}:
        num_experiments = 1

    print(f"\n" + "=" * 70)
    print(f"RUNNING BENCHMARK: Method={method} | Tier={tier_cap} | Problems {start_i} to {end_i - 1}")
    print(f"Hyperparameters: {best_config}")
    print("=" * 70)

    for row_idx in range(start_i, end_i):
        problem_id = f"{tier_cap}-{row_idx}"
        print(f"\n---> [{method}] Running Problem: {problem_id}")

        dataset_inter, X_extra, y_extra, orig_eq = load_problem_data(tier_cap, row_idx)

        # Initialize approach model with optimal config
        model = approach_cls.from_config(best_config)

        # Run training on interpolation set
        dataset_inter.evaluate_approach(model, num_experiments=num_experiments)

        # Extract learned expressions
        learned_exprs = model.sympy if isinstance(model.sympy, list) else [model.sympy]

        inter_mses, inter_r2s = [], []
        extra_mses, extra_r2s = [], []

        X_inter, y_inter = dataset_inter.X, dataset_inter.y

        # Evaluate each learned expression
        for expr in learned_exprs:
            i_mse, i_r2 = evaluate_expression(expr, X_inter, y_inter)
            e_mse, e_r2 = evaluate_expression(expr, X_extra, y_extra)

            inter_mses.append(i_mse)
            inter_r2s.append(i_r2)
            extra_mses.append(e_mse)
            extra_r2s.append(e_r2)

        # Compute summary statistics
        result_record = {
            "problem_id": problem_id,
            "tier": tier_cap,
            "method": method,
            "original_equation": orig_eq,
            "config": best_config,
            "num_experiments": num_experiments,
            "learned_expressions": [str(e) for e in learned_exprs],
            "raw_metrics": {
                "inter_mse": inter_mses,
                "inter_r2": inter_r2s,
                "extra_mse": extra_mses,
                "extra_r2": extra_r2s,
            },
            "summary_metrics": {
                "inter_mse_mean": float(np.nanmean(inter_mses)),
                "inter_mse_std": float(np.nanstd(inter_mses)),
                "inter_r2_mean": float(np.nanmean(inter_r2s)),
                "inter_r2_std": float(np.nanstd(inter_r2s)),
                "extra_mse_mean": float(np.nanmean(extra_mses)),
                "extra_mse_std": float(np.nanstd(extra_mses)),
                "extra_r2_mean": float(np.nanmean(extra_r2s)),
                "extra_r2_std": float(np.nanstd(extra_r2s)),
            },
        }

        # Save problem result to JSON
        res_file = out_dir / f"{problem_id}.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(result_record, f, indent=2, default=str)

        print(f"✓ Saved results to {res_file}")
        print(f"  Inter MSE Mean: {result_record['summary_metrics']['inter_mse_mean']:.6f}")
        print(f"  Extra MSE Mean: {result_record['summary_metrics']['extra_mse_mean']:.6f}")


if __name__ == "__main__":
    # Example execution: run Gplearn on Easy tier for problems 0 to 4
    run_benchmark(
        method="UDSR",
        tier="Hard",
        start_idx=0,
        end_idx=None,
    )