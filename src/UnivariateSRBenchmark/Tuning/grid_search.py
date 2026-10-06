"""
Generic grid search runner.
"""
import os

# Add MSYS2 core utilities (head.exe, cat.exe, rm.exe) to PATH
msys_usr_bin = r"C:\msys64\usr\bin"
git_usr_bin = r"C:\Program Files\Git\usr\bin"
if os.path.exists(msys_usr_bin):
    os.environ["PATH"] = msys_usr_bin + os.pathsep + os.environ["PATH"]
elif os.path.exists(git_usr_bin):
    os.environ["PATH"] = git_usr_bin + os.pathsep + os.environ["PATH"]

import re
import sys
import yaml
import json
import itertools
import importlib
import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from typing import Type
from SRToolkit.dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary
from typing import Any, Iterator, List

from UnivariateSRBenchmark.utils import sympy_to_srtoolkit_tokens, get_project_root
from UnivariateSRBenchmark.config import *

GRIDS_PATH = Path(__file__).parent / "grids.yaml"
GRID_RESULTS_PATH = Path(__file__).parent / "grid_results"


def load_grid(method: str, grids_path: Path = GRIDS_PATH) -> Dict[str, List[Any]]:
    with open(grids_path, "r") as f:
        all_grids = yaml.safe_load(f)
    if method not in all_grids:
        raise KeyError(f"No grid found for method '{method}'. "f"Available methods: {sorted(all_grids.keys())}")
    return all_grids[method]


def iter_configs(grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def get_approach_class(name: str) -> Type:
    if name not in APPROACH_REGISTRY:
        valid_methods = ", ".join(APPROACH_REGISTRY.keys())
        raise KeyError(f"Unknown approach '{name}'. Valid options: {valid_methods}")
    module_path, class_name = APPROACH_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_dataset_id(dataset_id: str):
    """ Parse strings like 'Easy-25' -> ('Easy', 25)."""
    match = re.match(r"^(Easy|Medium|Hard)-(\d+)$", dataset_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse dataset id '{dataset_id}'. "
                         f"Expected format '<Difficulty>-<row_index>', e.g. 'Easy-25'.")
    difficulty, row_str = match.groups()
    return difficulty.capitalize(), int(row_str)


def run_grid_search(
    method: str,
    dataset,
    num_experiments: int = 5,
    out_dir: str = None,
    grids_path: Path = GRIDS_PATH,
) -> List[Dict[str, Any]]:
    """
    Run a full grid search for `method` on `dataset`.
    - method: name matching a key in grids.yaml / APPROACH_REGISTRY, e.g. "Gplearn"
    - dataset: an SRToolkit dataset object with .evaluate_approach(...)
    """
    # target_version = "1.26.4"  # Just a reminder that DSR, UDSR, and BSR works on numpy 1.26.4 while the other methods don't
    # restore_version = "2.1.0"
    # if method in ['DSR', 'uDSR', 'BSR']:
    #     print(f"Downgrading numpy to {target_version}...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", f"numpy=={target_version}", "--quiet",])
    #     # install pip install setuptools==81.0.0

    grid = load_grid(method, grids_path)
    approach_cls = get_approach_class(method)

    if isinstance(dataset, str):
        dataset = build_custom_dataset(dataset)

    try:
        dataset_name = dataset.dataset_name
    except AttributeError:
        dataset_name = dataset.kwargs["dataset_name"]

    if out_dir is not None:
        out_path = Path(out_dir)
    else:
        out_path = GRID_RESULTS_PATH / dataset_name / method

    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    if method == 'AIFeynman':
        num_experiments = 1  # it's deterministic

    for i, cfg in enumerate(iter_configs(grid)):
        print(f"[{method}] run {i:04d} — config: {cfg}")

        model = approach_cls.from_config(cfg)
        dataset.evaluate_approach(model, num_experiments=num_experiments)

        non_nan_errors = [e for e in model.errors if not np.isnan(e)]
        min_err = min(non_nan_errors) if non_nan_errors else None
        record = {"dataset":dataset, "method": method, "run_index": i, "config": cfg, "SymPy": model.sympy,
                  "errors": model.errors, "min_error": min_err}
        all_results.append(record)

        with open(out_path / f"run_{i:04d}.json", "w") as f:
            json.dump(record, f, indent=2, default=str)

    # summary file with every run for convenience
    with open(out_path / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def build_custom_dataset(dataset_id: str) -> SR_dataset:
    """
    Build an SR_dataset from one row of the corresponding difficulty parquet file.
    """
    difficulty, row_index = _parse_dataset_id(dataset_id)

    parquet_path = Path(os.path.join(get_project_root(), DATASET_FILES.get(difficulty)))
    if parquet_path is None or not parquet_path.exists():
        raise FileNotFoundError(f"No parquet file found for difficulty '{difficulty}' at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    lengths = df["skeleton"].str.len()
    median_idx = (lengths - lengths.median()).abs().idxmin()

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} out of range for '{difficulty}' dataset "
                         f"(has {len(df)} rows).")

    row = df.iloc[row_index]

    equation_str = row["equation"]
    X = np.asarray(row["X"], dtype=float)
    y = np.asarray(row["Y"], dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    sl = SymbolLibrary.default_symbols(num_variables=n_features)

    # Try to convert the ground-truth equation string into SRToolkit tokens.
    ground_truth_tokens = None
    try:
        X_syms = {f"X{i}": sp.Symbol(f"X{i}", real=True) for i in range(n_features)}
        expr = sp.sympify(equation_str, locals=X_syms)
        expr = expr.subs({sp.Symbol(f"X{i}"): sp.Symbol(f"X_{i}") for i in range(n_features)})
        ground_truth_tokens, _ = sympy_to_srtoolkit_tokens(expr)
    except Exception as e:
        print(f"[build_custom_dataset] Warning: could not tokenize ground truth"
              f"'{equation_str}' ({e}). Proceeding with ground_truth=None." )

    dataset_name = f"{difficulty}-{row_index}"

    return SR_dataset(
        X=X,
        y=y,
        symbol_library=sl,
        dataset_name=dataset_name,
        original_equation=str(equation_str),
        ground_truth=ground_truth_tokens,
        max_evaluations=50000,
        success_threshold=1e-6,
        constant_bounds=(-100.0, 100.0),
        max_expr_length=150,
    )


if __name__ == "__main__":
    # # If an existing dataset within SRToolkit is used:
    # from SRToolkit.dataset import Nguyen
    # bm = Nguyen()
    # dtaset = bm.create_dataset("NG-12")

    # If one of the proposed datasets is used:
    # For the Easy tier, use 'Easy-9'
    # For the Medium tier, use 'Medium-0'
    # For the Hard tier, use 'Hard-9'
    dtaset = 'Hard-9'
    print(dtaset)
    results = run_grid_search("Gplearn", dtaset)
