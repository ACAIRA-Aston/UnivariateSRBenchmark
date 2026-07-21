"""
Generic grid search runner.
"""

import yaml
import json
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List


GRIDS_PATH = Path(__file__).parent / "grids.yaml"


def _get_registry() -> Dict[str, Any]:
    from src.Methods.gplearn_approach import GplearnApproach

    registry = {
        "Gplearn": GplearnApproach,
        # "Operon": OperonApproach,
        # "PySR": PySRApproach,
        # "GPGOMEA": GPGomeaApproach,
        # "MRGP": MRGPApproach,
        # "EPLEX": EplexApproach,
        # "FEAT": FeatApproach,
        # "ITEA": IteaApproach,
        # "FFX": FfxApproach,
        # "uDSR": UdsrApproach,
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

    out_path = Path(out_dir) if out_dir else Path(f"grid_results/{method}")
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, cfg in enumerate(iter_configs(grid)):
        print(f"[{method}] run {i:04d} — config: {cfg}")

        model = approach_cls.from_config(cfg)
        result = dataset.evaluate_approach(model, num_experiments=num_experiments)

        record = {"method": method, "run_index": i, "config": cfg, "result": result, "SymPy": model.sympy,
                  "errors": model.errors, "min_error": min(model.errors)}
        all_results.append(record)

        with open(out_path / f"run_{i:04d}.json", "w") as f:
            json.dump(record, f, indent=2, default=str)

    # summary file with every run for convenience
    with open(out_path / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Custom datasets ...TODO
# ---------------------------------------------------------------------------
def build_dataset(benchmark_name: str, dataset_name: str):
    pass


def main():
    parser = argparse.ArgumentParser(description="Run grid search for an SR approach.")
    parser.add_argument("--method", required=True, help="Method name, e.g. Gplearn")
    parser.add_argument("--benchmark", default="Nguyen", help="Benchmark suite name")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. NG-1")
    parser.add_argument("--num_experiments", type=int, default=1)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.benchmark, args.dataset)
    run_grid_search(
        method=args.method,
        dataset=dataset,
        num_experiments=args.num_experiments,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    from SRToolkit.dataset import Nguyen
    bm = Nguyen()
    dtaset = bm.create_dataset("NG-1")

    results = run_grid_search("Gplearn", dtaset)
