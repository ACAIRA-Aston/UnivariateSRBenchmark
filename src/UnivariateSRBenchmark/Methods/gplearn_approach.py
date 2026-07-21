import numpy as np
from dataclasses import dataclass
from typing import Optional
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [200, 500, 1000, 2000]
# generations: [20, 50, 100, 200]
# tournament_size: [5, 10, 20, 50]
# p_crossover: [0.6, 0.7, 0.8, 0.9]
# p_subtree_mutation: [0.05, 0.1, 0.15, 0.2]
# parsimony_coefficient: [0.0001, 0.001, 0.01, 0.1]


@dataclass
class GplearnConfig(ApproachConfig):
    name: str = "Gplearn"
    population_size: int = 500
    generations: int = 2
    tournament_size: int = 20
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    parsimony_coefficient: float = 0.001


class GplearnApproach(SR_approach):
    def __init__(self, population_size=500, generations=2, tournament_size=20, p_crossover=0.7, p_subtree_mutation=0.1,
                 parsimony_coefficient=0.001):
        self.cfg = GplearnConfig(population_size = population_size,
                                 generations = generations,
                                 tournament_size = tournament_size,
                                 p_crossover = p_crossover,
                                 p_subtree_mutation = p_subtree_mutation,
                                 parsimony_coefficient = parsimony_coefficient)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []


        def _protected_exp(x):
            return np.exp(np.clip(x, -100, 100))

        exp = make_function(function=_protected_exp, name='exp', arity=1)

        def _protected_cube(x):
            return np.power(np.clip(x, -1e100, 1e100), 3)

        cube = make_function(function=_protected_cube, name='cube', arity=1)

        def _protected_tan(x):
            with np.errstate(divide='ignore', invalid='ignore'):
                res = np.tan(x)
                # Catch asymptotes or infinite inputs turning into NaN/Inf
                return np.where(np.isnan(res) | np.isinf(res), 0.0, res)

        tan = make_function(function=_protected_tan, name='tan', arity=1)

        self.function_set = ['add', 'sub', 'mul', 'div', 'sin', tan, exp, 'neg', 'log', 'sqrt', 'abs', cube]

        self.converter = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            # protected division
            "div": lambda x, y: x / (y + 0.00001),
            "neg": lambda x: -x,
            "sin": sp.sin,
            "tan": sp.tan,
            "exp": lambda x: sp.exp(x),
            # gplearn protected sqrt
            "sqrt": lambda x: sp.sqrt(sp.Abs(x)),
            # gplearn protected log
            "log": lambda x: sp.log(sp.Abs(x)),
            "abs": sp.Abs,
            "cube": lambda x: x ** 3,
        }

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        model = SymbolicRegressor(
            population_size=self.cfg.population_size,
            generations=self.cfg.generations,
            tournament_size=self.cfg.tournament_size,
            p_crossover=self.cfg.p_crossover,
            p_subtree_mutation=self.cfg.p_subtree_mutation,
            p_hoist_mutation=0.05,
            p_point_mutation=0.05,
            parsimony_coefficient=self.cfg.parsimony_coefficient,
            function_set=self.function_set,
            random_state=seed,
            warm_start=False,
            verbose=True
        )
        model.fit(X, y)

        # Extract and evaluate expressions
        program = model._program
        if program is not None and not sr_evaluator.should_stop:
            X_syms = {f'X0':sp.Symbol('X0', real=True)}
            expr = sp.sympify(str(program), locals={**self.converter, **X_syms})
            # gplearn uses X0, X1... -> SRToolkit wants X_0, X_1...
            expr = expr.subs(X_syms['X0'], sp.Symbol(f'X_0'))

            f = sp.lambdify(sp.Symbol('X_0'), expr, modules="numpy")
            values = f(X)
            try:
                tokens, const_values = sympy_to_srtoolkit_tokens(expr)
                sr_evaluator.evaluate_expr(tokens)
                err = np.sqrt(np.mean((values[:, 0] - y)**2))
                for key in sr_evaluator.models.keys():
                    sr_evaluator.models[key].error = err
                self.errors.append(err)
                self.sympy.append(expr)
            except ValueError:
                print("Iteration omitted")

    @classmethod
    def from_config(self, config: dict) -> "GplearnApproach":
        return self(
            population_size=config.get("population_size", 500),
            generations=config.get("generations", 300),
            tournament_size=config.get("tournament_size", 20),
            p_crossover=config.get("p_crossover", 0.7),
            p_subtree_mutation=config.get("p_subtree_mutation", 0.1),
            parsimony_coefficient=config.get("parsimony_coefficient", 0.001),
        )
