import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Mapping difficulty tiers to the default dataset names used in the grid search
DIFFICULTY_MAP = {
    "Easy": "Easy-9",
    "Medium": "Medium-0",
    "Hard": "Hard-9",
}

# Base path where grid results are stored
GRID_RESULTS_PATH = Path(__file__).parent / "grid_results"


def parse_error(val: Any) -> float:
    """Safely converts min_error to a float for comparison, returning infinity for invalid/NaN values."""
    if val is None or val in ("None", "null", "NaN"):
        return float("inf")
    try:
        f_val = float(val)
        return float("inf") if f_val != f_val else f_val  # NaN check
    except (ValueError, TypeError):
        return float("inf")


def rebuild_summary_and_get_best(
    method: str,
    difficulty: str,
    grid_results_base: Path = GRID_RESULTS_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Reads all run_*.json files for a given method and difficulty tier, rebuilds summary.json,
    and returns the record with the best hyperparameter configuration.
    """
    diff_key = difficulty.strip()
    dataset_folder = DIFFICULTY_MAP.get(diff_key, difficulty.strip())

    run_dir = grid_results_base / dataset_folder / method

    if not run_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {run_dir}")

    # Find all run JSON files
    run_files = [f for f in run_dir.glob("run_*.json") if f.name != "summary.json"]
    if not run_files:
        raise FileNotFoundError(f"No run_*.json files found in {run_dir}")

    # Sort files numerically
    def get_run_index(p: Path) -> int:
        match = re.search(r"run_(\d+)\.json$", p.name)
        return int(match.group(1)) if match else -1

    run_files.sort(key=get_run_index)

    # Load each JSON file
    all_results = []
    for file_path in run_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
                all_results.append(record)
        except Exception as e:
            print(f"[Warning] Could not parse {file_path.name}: {e}")

    if not all_results:
        print(f"No valid records loaded from {run_dir}")
        return None

    # Overwrite / Create the updated summary.json file
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"✓ Created {summary_path} containing {len(all_results)} runs.")

    # Find the best configuration (minimal min_error; ties broken by first occurrence)
    best_record = None
    best_error = float("inf")

    for record in all_results:
        err = parse_error(record.get("min_error"))
        if err < best_error:
            best_error = err
            best_record = record

    # If all runs produced NaN / Inf errors, fall back to the first configuration
    if best_record is None:
        best_record = all_results[0]
        print("[Notice] No runs converged with a valid error. Defaulting to first configuration.")

    print("\n" + "=" * 60)
    print(f"BEST CONFIGURATION SUMMARY ({method} on {dataset_folder})")
    print("=" * 60)
    print(f"Run Index:      {best_record.get('run_index')}")
    print(f"Min Error:      {best_record.get('min_error')}")
    print(f"SymPy Expr:     {best_record.get('SymPy')}")
    print("Best Hyperparameters:")
    print(json.dumps(best_record.get("config"), indent=2))
    print("=" * 60 + "\n")

    # Save the best record to a file named 'best_run.json' in the same folder
    best_out_path = run_dir / "best_run.json"
    with open(best_out_path, "w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2, default=str)

    print(f"✓ Saved best run details to {best_out_path}")

    return best_record


if __name__ == "__main__":
    # Example usage: specify method and difficulty tier ("Easy", "Medium", or "Hard")
    method_name = "GPGOMEA"
    difficulty_tier = "Hard"

    try:
        best_run = rebuild_summary_and_get_best(
            method=method_name,
            difficulty=difficulty_tier
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")