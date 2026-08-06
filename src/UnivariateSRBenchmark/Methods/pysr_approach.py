import numpy as np
from dataclasses import dataclass
from typing import Optional
from pysr import PySRRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [30, 100, 250, 500]
# ncycles_per_iteration: [200, 500, 1000, 2000]
# maxsize: [15, 20, 25, 30]
# maxdepth: [5, 8, 12, 15]
# parsimony: [0.0001, 0.001, 0.005, 0.01]
# niterations: [50, 100, 200, 500]


@dataclass
class PySRConfig(ApproachConfig):
    name: str = "PySR"
    population_size: int = 100
    ncycles_per_iteration: int = 500
    maxsize: int = 20
    maxdepth: int = 8
    parsimony: float = 0.001
    niterations: int = 100


class PySRApproach(SR_approach):
    def __init__(self, population_size=100, ncycles_per_iteration=500, maxsize=20, maxdepth=8,
                 parsimony=0.001, niterations=100):
        self.cfg = PySRConfig(population_size=population_size,
                              ncycles_per_iteration=ncycles_per_iteration,
                              maxsize=maxsize,
                              maxdepth=maxdepth,
                              parsimony=parsimony,
                              niterations=niterations)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.binary_operators = ["+", "-", "*", "/"]
        self.unary_operators = [
            # Trigonometric
            "sin",
            "tan",
            "asin",
            "atan",
            # Hyperbolic
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            # Exponential & Logarithmic
            "exp",
            "log",
            # Basic Arithmetic
            "sqrt",
            "abs",
            # Algebraic Structural Ops
            "square",
            "cube",
        ]

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        model = PySRRegressor(
            population_size=self.cfg.population_size,
            ncycles_per_iteration=self.cfg.ncycles_per_iteration,
            maxsize=self.cfg.maxsize,
            maxdepth=self.cfg.maxdepth,
            parsimony=self.cfg.parsimony,
            niterations=self.cfg.niterations,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            random_state=seed,
            deterministic=seed is not None,
            parallelism="serial" if seed is not None else "multithreading",
            progress=False,
            verbosity=0,
        )
        # variable_names is a fit()-time argument in PySR, not a constructor one
        model.fit(X, y, variable_names=["X_0"])

        if not sr_evaluator.should_stop:
            expr = model.sympy()
            if 'X_0' not in str(expr.free_symbols):
                if expr.is_real:
                    values = np.repeat(float(expr), len(X), axis=0)[:, None]
                else:
                    values = np.repeat(1000, len(X), axis=0)[:, None]  # Penalize these cases
            else:
                f = sp.lambdify(sp.Symbol('X_0'), expr, modules="numpy")
                values = f(X)
            try:
                tokens, const_values = sympy_to_srtoolkit_tokens(expr)
                err = np.sqrt(np.mean((values[:, 0] - y) ** 2))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=err)
                self.errors.append(err)
                self.sympy.append(expr)
            except ValueError:
                print("Iteration omitted")
                tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                self.errors.append(1000)
                self.sympy.append(sp.sympify('0'))

    @classmethod
    def from_config(self, config: dict) -> "PySRApproach":
        return self(
            population_size=config.get("population_size", 100),
            ncycles_per_iteration=config.get("ncycles_per_iteration", 500),
            maxsize=config.get("maxsize", 20),
            maxdepth=config.get("maxdepth", 8),
            parsimony=config.get("parsimony", 0.001),
            niterations=config.get("niterations", 100),
        )
