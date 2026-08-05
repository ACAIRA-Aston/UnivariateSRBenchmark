import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from taylorGP.genetic import SymbolicRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult
from taylorGP.functions import make_function

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [100, 200, 500, 1000]
# generations: [20, 50, 100, 200]
# crossover_rate: [0.6, 0.7, 0.8, 0.9]
# mutation_rate: [0.05, 0.1, 0.15, 0.2]


@dataclass
class TaylorGPConfig(ApproachConfig):
    name: str = "TaylorGP"
    population_size: int = 200
    generations: int = 100
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1


class TaylorGPApproach(SR_approach):
    def __init__(self, population_size=200, generations=100, crossover_rate=0.7, mutation_rate=0.1):
        self.cfg = TaylorGPConfig(population_size=population_size,
                                  generations=generations,
                                  crossover_rate=crossover_rate,
                                  mutation_rate=mutation_rate)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        def _protected_exp(x):
            # Clip to avoid overflow/underflow floating point errors
            return np.exp(np.clip(x, -80, 80))

        def _protected_asin(x):
            # Domain: [-1, 1]
            return np.arcsin(np.clip(x, -1.0, 1.0))

        def _atan(x):
            # Domain: (-inf, inf)
            return np.arctan(x)

        def _protected_sinh(x):
            # Clip to avoid overflow in sinh
            return np.sinh(np.clip(x, -80, 80))

        def _protected_cosh(x):
            # Clip to avoid overflow in cosh
            return np.cosh(np.clip(x, -80, 80))

        def _tanh(x):
            # Domain: (-inf, inf), Output: [-1, 1]
            return np.tanh(x)

        def _asinh(x):
            # Domain: (-inf, inf)
            return np.arcsinh(x)

        def _protected_acosh(x):
            # Domain: [1, inf)
            return np.arccosh(np.maximum(1.0, x))

        def _protected_atanh(x):
            # Domain: (-1, 1)
            return np.arctanh(np.clip(x, -0.999, 0.999))

        exp_func = make_function(function=_protected_exp, name='exp', arity=1)
        asin_func = make_function(function=_protected_asin, name='asin', arity=1)
        atan_func = make_function(function=_atan, name='atan', arity=1)
        sinh_func = make_function(function=_protected_sinh, name='sinh', arity=1)
        cosh_func = make_function(function=_protected_cosh, name='cosh', arity=1)
        tanh_func = make_function(function=_tanh, name='tanh', arity=1)
        asinh_func = make_function(function=_asinh, name='asinh', arity=1)
        acosh_func = make_function(function=_protected_acosh, name='acosh', arity=1)
        atanh_func = make_function(function=_protected_atanh, name='atanh', arity=1)

        self.function_set = [# Built-in TaylorGP string operations:
                                'add', 'sub', 'mul', 'div', 'sin', 'tan', 'log', 'sqrt', 'abs',

                                # Custom wrapped objects:
                                exp_func,
                                asin_func,
                                atan_func,
                                sinh_func,
                                cosh_func,
                                tanh_func,
                                asinh_func,
                                acosh_func,
                                atanh_func
                            ]

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        model = SymbolicRegressor(population_size=self.cfg.population_size, init_depth=(2, 5),
                        generations=self.cfg.generations, stopping_criteria=1e-10,
                        function_set=self.function_set,
                        p_crossover=self.cfg.crossover_rate, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=self.cfg.mutation_rate,
                        max_samples=1.0, verbose=1,
                        parsimony_coefficient=0.1,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=42, low_memory=True,
                        max_time = 3600-10)
        model.fit(X, y)

        try:
            expr = model.sympy_global_best.subs(sp.Symbol('x0'), sp.Symbol('X_0'))
        except Exception:
            print("Iteration omitted")
            return

        print('Expr', expr)
        if 'X_0' not in str(expr.free_symbols):
            if expr.is_real:
                values = np.repeat(float(expr), len(X), axis=0)[:, None]
            else:
                values = np.repeat(1000, len(X), axis=0)[:, None]  # Penalize these cases
        else:
            f = sp.lambdify(sp.Symbol('X_0'), expr, modules="numpy")
            values = f(X)
            values = np.asarray(values).reshape(-1, 1) if np.ndim(values) < 2 else values
        try:
            tokens, const_values = sympy_to_srtoolkit_tokens(expr)
            err = np.sqrt(np.mean((values[:, 0] - y) ** 2))
            sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=err)
            self.errors.append(err)
            self.sympy.append(expr)
        except ValueError:
            print("Iteration omitted")


    @classmethod
    def from_config(self, config: dict) -> "TaylorGPApproach":
        return self(
            population_size=config.get("population_size", 250),
            generations=config.get("generations", 100),
            crossover_rate=config.get("crossover_rate", 0.7),
            mutation_rate=config.get("mutation_rate", 0.1),
        )
