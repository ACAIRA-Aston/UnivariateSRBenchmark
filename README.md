[![ExCAUS Dataset](https://img.shields.io/badge/SR_Benchmark-ExCAUS_Dataset-blue?logo=github)](https://github.com/ACAIRA-Aston/SRSkeletons)
[![arXiv](https://img.shields.io/badge/arXiv-2610.20108-b31b1b.svg)](https://arxiv.org/abs/2610.20108)

# ExCAUS-Bench


**ExCAUS-Bench** is a benchmark for evaluating symbolic regression (SR) methods on challenging univariate problems. It contains 150 problems divided into Easy, Medium, and Hard tiers, together with results from 13 SR methods. Each problem has interpolation data for fitting and a separate extrapolation set for testing whether a recovered expression generalizes beyond its fitting range. This makes the benchmark useful not only for comparing in-domain fit, but also for studying how reliably SR methods recover functions over wider domains.

The benchmark is the evaluation counterpart to [SRSkeletons](https://github.com/ACAIRA-Aston/SRSkeletons), which contains the code for creating the **ExCAUS dataset**. This repository focuses on the benchmark problems, evaluation workflow, and results obtained on them.

## Project Structure

| Path | Purpose |
| --- | --- |
| `HyperparameterTuning.py` | Example entry point for grid-search tuning on one selected problem. |
| `MainRun.py` | Example entry point for running a method across a tier. |
| `src/UnivariateSRBenchmark/Datasets/` | Easy, Medium, and Hard problem tables, plus their paired interpolation and extrapolation tables. |
| `src/UnivariateSRBenchmark/Methods/` | Adapters and bundled implementations for the SR methods. |
| `src/UnivariateSRBenchmark/Tuning/grid_search.py` | Loads parameter grids, builds an `SR_dataset`, evaluates configurations, and writes per-run results. |
| `src/UnivariateSRBenchmark/Tuning/grids.yaml` | Hyperparameter values searched for each supported method. |
| `src/UnivariateSRBenchmark/RunBench/run_benchmark.py` | Runs a method with the selected best configuration and evaluates interpolation and extrapolation performance. |
| `src/UnivariateSRBenchmark/RunBench/benchmark_results/` | Per-problem JSON outputs organized by tier and method. |

## Hyperparameter Tuning

`run_grid_search()` evaluates every configuration in `grids.yaml` for a method on one problem. The root-level script demonstrates the call:

```python
from src.UnivariateSRBenchmark.Tuning.grid_search import run_grid_search

results = run_grid_search(
    method="UDSR",
    dataset="Hard-9",
    num_experiments=5,
    out_dir=None,
)
```

Parameters:

- `method`: Method name matching a key in `APPROACH_REGISTRY` and `grids.yaml`, such as `"UDSR"`.
- `dataset`: An `SR_dataset` instance or a problem ID such as `"Easy-9"`, `"Medium-0"`, or `"Hard-9"`.
- `num_experiments`: Number of evaluations per grid configuration; defaults to `5`.
- `out_dir`: Optional output directory. By default, tuning output is written under `src/UnivariateSRBenchmark/Tuning/grid_results/<problem-id>/<method>/`.
- `grids_path`: Optional path to a YAML grid; defaults to `src/UnivariateSRBenchmark/Tuning/grids.yaml`.

After tuning, call `rebuild_summary_and_get_best(method, difficulty)` from `src/UnivariateSRBenchmark/Tuning/optimal_hyperparameters.py` to select the lowest-error configuration and write `best_run.json`. Benchmark runs use this saved configuration for the tier's default tuning problem (`Easy-9`, `Medium-0`, or `Hard-9`).

## Run the Benchmark

`run_benchmark()` evaluates the selected method on one tier using its saved best configuration. For example, this runs all Easy-tier problems:

```python
from src.UnivariateSRBenchmark.RunBench.run_benchmark import run_benchmark

run_benchmark(
    method="NeSymReS",
    tier="Easy",
    start_idx=0,
    end_idx=None,
    num_experiments=10,
)
```

Parameters:

- `method`: Registered method name, for example `"NeSymReS"`.
- `tier`: Difficulty tier, `"Easy"`, `"Medium"`, or `"Hard"`.
- `start_idx`: First zero-based problem row to run; defaults to the beginning of the tier.
- `end_idx`: Last zero-based problem row to run, inclusive. `None` runs through the last row.
- `num_experiments`: Number of attempts per problem; defaults to `10`. AIFeynman and DSR are forced to one run because they are treated as deterministic by this runner.

Each problem's JSON result is written to `src/UnivariateSRBenchmark/RunBench/benchmark_results/<Tier>/<Method>/<Tier>-<index>.json`. It includes the target equation, configuration, learned expressions, and mean/std MSE and $R^2$ for both interpolation and extrapolation data.

## Benchmark Datasets

The parquet files are in `src/UnivariateSRBenchmark/Datasets/`:

| Tier | Problem table | Interpolation samples | Extrapolation samples |
| --- | --- | --- | --- |
| Easy | `easy.parquet` | `interpolation_easy.parquet` | `extrapolation_easy.parquet` |
| Medium | `medium.parquet` | `interpolation_medium.parquet` | `extrapolation_medium.parquet` |
| Hard | `hard.parquet` | `interpolation_hard.parquet` | `extrapolation_hard.parquet` |

Each tier contains 50 problems (150 total). Rows correspond across the problem, interpolation, and extrapolation files: row `i` is problem `<Tier>-<i>` in all three. The problem tables contain these fields:

| Column | Description |
| --- | --- |
| `skeleton` | Canonical structural representation associated with the problem. |
| `equation` | Concrete target equation, represented as a string. |
| `X` | Input sample values stored as an array; univariate samples are reshaped to one feature when loaded. |
| `Y` | Corresponding target values stored as an array. |

The interpolation and extrapolation parquet files use `equation`, `X`, and `Y` to provide the paired sample sets. The benchmark fits on interpolation `(X,Y)` and evaluates the learned expression on both sets. The easy, medium, and hard tiers provide increasing problem difficulty; extrapolation data tests behavior outside the fitting samples.

## Citation

If you use this benchmark or its results, please cite the ExCAUS work and acknowledge the benchmark repository:

```bibtex
@inproceedings{ExCAUS,
  title={Exhaustive Canonical Skeletons Generation: A Formal Dataset for Decomposable Symbolic Regression (in review)},
  author={Morales, Giorgio and Ek\'{a}rt, Anik\'{o}},
  volume={},
  number={},
  journal={IEEE Computational Intelligence Letters},
  year={2026},
  month={},
  pages={},
  doi={}
}
```

Dataset generation code: [ACAIRA-Aston/SRSkeletons](https://github.com/ACAIRA-Aston/SRSkeletons). Benchmark code and results: [ACAIRA-Aston/UnivariateSRBenchmark](https://github.com/ACAIRA-Aston/UnivariateSRBenchmark).
