import os
import sys

# Path to your compiled Fortran executables
aifeynman_code_dir = r"C:\Users\gsjj1\PycharmProjects\UnivariateSRBenchmark\src\UnivariateSRBenchmark\Methods\AIFeynman\Code"

# Prepend to environment PATH so Windows subprocesses can find the binaries
os.environ["PATH"] = aifeynman_code_dir + os.pathsep + os.environ.get("PATH", "")

# Optionally add to Python sys.path if module imports are failing
if aifeynman_code_dir not in sys.path:
    sys.path.insert(0, aifeynman_code_dir)

from UnivariateSRBenchmark.Methods.AIFeynman.Code.S_run_aifeynman import run_aifeynman
import gc
import tempfile
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Optional

from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import sympy_to_srtoolkit_tokens
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")


@dataclass
class AIFeynmanConfig(ApproachConfig):
    name: str = "AIFeynman"
    BF_try_time: int = 60           # Max search time (in seconds) per brute-force level
    BF_ops_file_type: str = "19ops.txt"  # Operator set ('7ops.txt', '10ops.txt', '14ops.txt', '19ops.txt')
    polyfit_deg: int = 4            # Maximum polynomial degree for polyfit step
    NN_epochs: int = 1000           # Neural network training epochs for symmetry/separability detection
    test_percentage: int = 0        # 0 because SRToolkit handles train/test evaluation separately


class AIFeynmanApproach(SR_approach):
    def __init__(
        self,
        BF_try_time: int = 60,
        BF_ops_file_type: str = "14ops.txt",
        polyfit_deg: int = 4,
        NN_epochs: int = 1000,
        test_percentage: int = 0,
    ):
        self.cfg = AIFeynmanConfig(
            BF_try_time=BF_try_time,
            BF_ops_file_type=BF_ops_file_type,
            polyfit_deg=polyfit_deg,
            NN_epochs=NN_epochs,
            test_percentage=test_percentage,
        )
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        if seed is not None:
            np.random.seed(seed)

        # Reshape input arrays to (N, 1) if necessary
        X_arr = X.reshape(-1, 1) if X.ndim == 1 else X
        y_arr = y.reshape(-1, 1) if y.ndim == 1 else y

        # Create temporary working directory for AI Feynman file I/O
        with tempfile.TemporaryDirectory() as temp_dir:
            pathdir = os.path.abspath(temp_dir) + os.sep
            filename = "sr_data.txt"

            # Combine X and y into a single matrix [X, y]
            data = np.hstack((X_arr, y_arr))
            np.savetxt(pathdir + filename, data)

            # Ensure AI Feynman's output results directory exists in CWD
            results_dir = os.path.abspath("results")
            os.makedirs(results_dir, exist_ok=True)

            try:
                run_aifeynman(
                    pathdir=pathdir,
                    filename=filename,
                    BF_try_time=self.cfg.BF_try_time,
                    BF_ops_file_type=self.cfg.BF_ops_file_type,
                    polyfit_deg=self.cfg.polyfit_deg,
                    NN_epochs=self.cfg.NN_epochs,
                    test_percentage=self.cfg.test_percentage,
                )
            except Exception as e:
                print(f"AI Feynman execution encountered an issue: {e}")

            # Collect generated Pareto expressions from AI Feynman output files
            results_file = os.path.join(results_dir, f"solution_{filename}")
            expressions = []

            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Split at most 5 times to separate the 5 metric columns,
                            # keeping the full expression in the 6th element
                            parts = line.split(maxsplit=5)
                            if len(parts) == 6:
                                expressions.append(parts[-1])


            # Parse expressions and evaluate them in SRToolkit
            if not sr_evaluator.should_stop:
                try:
                    expr = sp.sympify(expressions[-1])
                    for var_name in ["x0", "x1", "x_0", "x_1"]:
                        expr = expr.subs(sp.Symbol(var_name), sp.Symbol("X_0"))

                    if "X_0" not in str(expr.free_symbols):
                        if expr.is_real:
                            values = np.repeat(float(expr), len(X_arr), axis=0)[:, None]
                        else:
                            values = np.repeat(1000.0, len(X_arr), axis=0)[:, None]
                    else:
                        f = sp.lambdify(sp.Symbol("X_0"), expr, modules="numpy")
                        values = f(X_arr)
                    err = float(np.sqrt(np.mean((values - y_arr) ** 2)))
                    tokens, _ = sympy_to_srtoolkit_tokens(expr)
                    sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=err)
                    self.errors.append(err)
                    self.sympy.append(expr)
                except ValueError:
                    print("Iteration omitted")
                    tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify("0"))
                    sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                    self.errors.append(1000)
                    self.sympy.append(sp.sympify("0"))

                # Clean up generated solution files
                if os.path.exists(results_file):
                    try:
                        os.remove(results_file)
                    except OSError:
                        pass

        gc.collect()

    @classmethod
    def from_config(cls, config: dict) -> "AIFeynmanApproach":
        return cls(
            BF_try_time=config.get("BF_try_time", 60),
            BF_ops_file_type=config.get("BF_ops_file_type", "14ops.txt"),
            polyfit_deg=config.get("polyfit_deg", 4),
            NN_epochs=config.get("NN_epochs", 1000),
            test_percentage=config.get("test_percentage", 0),
        )