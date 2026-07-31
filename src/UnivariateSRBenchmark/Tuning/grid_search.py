"""
Generic grid search runner.
"""
import os
import re
import yaml
import json
import argparse
import itertools
import SRToolkit
import numpy as np
import sympy as sp
import pandas as pd
from pathlib import Path
from SRToolkit.dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary
from typing import Any, Dict, Iterator, List

from UnivariateSRBenchmark.utils import sympy_to_srtoolkit_tokens, get_project_root


GRIDS_PATH = Path(__file__).parent / "grids.yaml"

DATASET_FILES = {
    "Easy": Path("UnivariateSRBenchmark/Datasets/easy.parquet"),
    "Medium": Path("UnivariateSRBenchmark/Datasets/medium.parquet"),
    "Hard": Path("UnivariateSRBenchmark/Datasets/hard.parquet"),
}


def _get_registry() -> Dict[str, Any]:
    from UnivariateSRBenchmark.Methods.gplearn_approach import GplearnApproach
    from UnivariateSRBenchmark.Methods.pysr_approach import PySRApproach
    from UnivariateSRBenchmark.Methods.dsr_approach import DSRApproach
    from UnivariateSRBenchmark.Methods.udsr_approach import UDSRApproach
    from UnivariateSRBenchmark.Methods.e2e_approach import E2EApproach
    from UnivariateSRBenchmark.Methods.operon_approach import OperonApproach

    registry = {
        "Gplearn": GplearnApproach,
        "Operon": OperonApproach,
        "PySR": PySRApproach,
        # "GPGOMEA": GPGomeaApproach,
        # "MRGP": MRGPApproach,
        # "EPLEX": EplexApproach,
        # "FEAT": FeatApproach,
        # "ITEA": IteaApproach,
        "DSR": DSRApproach,
        "UDSR": UDSRApproach,
        "E2E": E2EApproach,
        # "TPSR": TpsrApproach,
        # "TaylorGP": TaylorGPApproach,
        # "SBPGP": SbpGpApproach,
        # "BSR": BsrApproach,
        # "NeSymReS": NeSymReSApproach,
    }
    return registry


def load_grid(method: str, grids_path: Path = GRIDS_PATH) -> Dict[str, List[Any]]:
    with open(grids_path, "r") as f:
        all_grids = yaml.safe_load(f)

    if method not in all_grids:
        raise KeyError(
            f"No grid found for method '{method}'. "
            f"Available methods: {sorted(all_grids.keys())}"
        )
    return all_grids[method]


def iter_configs(grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def get_approach_class(method: str):
    registry = _get_registry()
    if method not in registry:
        raise KeyError(
            f"No Approach class registered for method '{method}'. "
            f"Registered methods: {sorted(registry.keys())}"
        )
    return registry[method]


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
    grid = load_grid(method, grids_path)
    approach_cls = get_approach_class(method)

    if isinstance(dataset, str):
        dataset = build_custom_dataset(dataset)

    (Path(out_dir) if out_dir else Path(f"grid_results/{dataset.dataset_name}")).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) if out_dir else Path(f"grid_results/{dataset.dataset_name}/{method}")
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, cfg in enumerate(iter_configs(grid)):
        print(f"[{method}] run {i:04d} — config: {cfg}")

        model = approach_cls.from_config(cfg)
        dataset.evaluate_approach(model, num_experiments=num_experiments)

        record = {"dataset":dataset, "method": method, "run_index": i, "config": cfg, "SymPy": model.sympy,
                  "errors": model.errors, "min_error": min(model.errors)}
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
    dtaset = 'Easy-9'
    results = run_grid_search("Operon", dtaset)
