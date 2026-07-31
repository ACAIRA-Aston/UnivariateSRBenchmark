import numpy as np
from dataclasses import dataclass
from typing import Optional
from pyoperon.sklearn import SymbolicRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [250, 500, 1000, 2000]
# allowed_evaluations: [50000, 100000, 250000, 500000]
# crossover_probability: [0.6, 0.7, 0.8, 0.9]
# mutation_probability: [0.1, 0.15, 0.2, 0.25]
# max_length: [10, 15, 25, 50]


@dataclass
class OperonConfig(ApproachConfig):
    name: str = "Operon"
    population_size: int = 500
    allowed_evaluations: int = 100000
    crossover_probability: float = 0.7
    mutation_probability: float = 0.15
    max_length: int = 25


class OperonApproach(SR_approach):
    def __init__(self, population_size=500, allowed_evaluations=100000, crossover_probability=0.7,
                 mutation_probability=0.15, max_length=25):
        self.cfg = OperonConfig(population_size=population_size,
                                allowed_evaluations=allowed_evaluations,
                                crossover_probability=crossover_probability,
                                mutation_probability=mutation_probability,
                                max_length=max_length)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.allowed_symbols = ("add,sub,mul,div,sin,tan,exp,log,sqrt,abs,constant,variable,square,acos,asin,atan,"
                                "sinh,cosh,tanh")

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = np.ascontiguousarray(sr_evaluator.X, dtype=np.float32)
        y = np.ascontiguousarray(sr_evaluator.y, dtype=np.float32).ravel()

        model = SymbolicRegressor(
            allowed_symbols=self.allowed_symbols,
            population_size=self.cfg.population_size,
            max_evaluations=self.cfg.allowed_evaluations,
            crossover_probability=self.cfg.crossover_probability,
            mutation_probability=self.cfg.mutation_probability,
            max_length=self.cfg.max_length,
            generations=1_000_000,  # let allowed_evaluations be the effective budget
            n_threads=1,
            random_state=seed,
        )
        model.fit(X, y)

        if not sr_evaluator.should_stop:
            # get_model_string returns Operon's infix notation, which uses
            # '^' for powers (from the 'square' primitive) instead of '**'.
            expr_str = model.get_model_string(model.model_, precision=15, names=["X0"])
            expr_str = expr_str.replace("^", "**")
            X_syms = {"X0": sp.Symbol("X0", real=True)}
            expr = sp.sympify(expr_str, locals={**X_syms})
            expr = expr.subs(X_syms["X0"], sp.Symbol("X_0"))

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

    @classmethod
    def from_config(self, config: dict) -> "OperonApproach":
        return self(
            population_size=config.get("population_size", 500),
            allowed_evaluations=config.get("allowed_evaluations", 100000),
            crossover_probability=config.get("crossover_probability", 0.7),
            mutation_probability=config.get("mutation_probability", 0.15),
            max_length=config.get("max_length", 25),
        )
