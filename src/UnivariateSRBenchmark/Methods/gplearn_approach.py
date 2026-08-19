import numpy as np
from dataclasses import dataclass
from typing import Optional
import multiprocessing as mp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [200, 500, 1000, 2000]
# generations: [20, 50, 100, 200]
# tournament_size: [5, 10, 20, 50]
# p_crossover: [0.6, 0.7, 0.8, 0.9]
# p_subtree_mutation: [0.05, 0.1, 0.15, 0.2]
# parsimony_coefficient: [0.0001, 0.001, 0.01, 0.1]
import time


def _search_worker(cfg, function_set, X, y, seed, queue):
    try:
        model = SymbolicRegressor(
            population_size=cfg.population_size,
            generations=cfg.generations,
            tournament_size=cfg.tournament_size,
            p_crossover=cfg.p_crossover,
            p_subtree_mutation=cfg.p_subtree_mutation,
            p_hoist_mutation=0.05,
            p_point_mutation=0.05,
            parsimony_coefficient=cfg.parsimony_coefficient,
            function_set=function_set,
            random_state=seed,
            warm_start=False,
            verbose=False
        )

        start = time.time()
        model.fit(X, y)
        print("Time elapsed fit: ", time.time() - start)

        converter = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            # protected division
            "div": lambda x, y: x / (y + 0.00001),
            "neg": lambda x: -x,
            "sin": sp.sin,
            "tan": sp.tan,
            "arcsin": sp.asin,
            "arctan": sp.atan,
            # Hyperbolic
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "arsinh": sp.asinh,
            "arcosh": sp.acosh,
            "artanh": sp.atanh,
            "exp": lambda x: sp.exp(x),
            # gplearn protected sqrt
            "sqrt": lambda x: sp.sqrt(sp.Abs(x)),
            # gplearn protected log
            "log": lambda x: sp.log(sp.Abs(x)),
            "abs": sp.Abs,
            "cube": lambda x: x ** 3,
        }

        program = model._program
        if program is not None:
            start = time.time()
            X_syms = {'X0': sp.Symbol('X0', real=True)}
            expr = sp.sympify(str(program), locals={**converter, **X_syms})
            expr = expr.subs(X_syms['X0'], sp.Symbol('X_0'))

            if 'X_0' not in str(expr.free_symbols):
                if expr.is_real:
                    values = np.repeat(float(expr), len(X), axis=0)[:, None]
                else:
                    values = np.repeat(1000, len(X), axis=0)[:, None]
            else:
                f = sp.lambdify(sp.Symbol('X_0'), expr, modules="numpy")
                values = f(X)
            print("Time processing: ", time.time() - start)

            try:
                start = time.time()
                tokens, const_values = sympy_to_srtoolkit_tokens(expr)
                err = np.sqrt(np.mean((values[:, 0] - y) ** 2))
                print("Time evaluation: ", time.time() - start)

                queue.put({"tokens": tokens, "err": err, "expr": expr})
                return
            except ValueError:
                pass

        # Failure fallback inside execution
        tokens, _ = sympy_to_srtoolkit_tokens(sp.sympify('0'))
        queue.put({"tokens": tokens, "err": 1000, "expr": sp.sympify('0')})

    except Exception as e:
        print(f"Iteration omitted due to error: {e}")
        tokens, _ = sympy_to_srtoolkit_tokens(sp.sympify('0'))
        queue.put({"tokens": tokens, "err": 1000, "expr": sp.sympify('0')})

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

        def _protected_arcsin(x):
            return np.arcsin(np.clip(x, -1.0, 1.0))

        arcsin = make_function(function=_protected_arcsin, name='arcsin', arity=1)

        def _protected_arctan(x):
            return np.arctan(x)

        arctan = make_function(function=_protected_arctan, name='arctan', arity=1)

        # --- Hyperbolic ---
        def _protected_sinh(x):
            return np.sinh(np.clip(x, -100, 100))

        sinh = make_function(function=_protected_sinh, name='sinh', arity=1)

        def _protected_cosh(x):
            return np.cosh(np.clip(x, -100, 100))

        cosh = make_function(function=_protected_cosh, name='cosh', arity=1)

        def _protected_tanh(x):
            return np.tanh(x)

        tanh = make_function(function=_protected_tanh, name='tanh', arity=1)

        def _protected_arsinh(x):
            return np.arcsinh(x)

        arsinh = make_function(function=_protected_arsinh, name='arsinh', arity=1)

        def _protected_arcosh(x):
            return np.arccosh(1.0 + np.abs(x))

        arcosh = make_function(function=_protected_arcosh, name='arcosh', arity=1)

        def _protected_artanh(x):
            # Domain (-1, 1) -> clip to [-0.999, 0.999]
            return np.arctanh(np.clip(x, -0.999, 0.999))

        artanh = make_function(function=_protected_artanh, name='artanh', arity=1)

        self.function_set = ['add', 'sub', 'mul', 'div', 'sin', tan, exp, 'neg', 'log', 'sqrt', 'abs', cube,
                             arcsin, arctan, sinh, cosh, tanh, arsinh, arcosh, artanh]

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None, timeout: float = 1000.0):
        if sr_evaluator.should_stop:
            return

        X = sr_evaluator.X
        y = sr_evaluator.y

        ct = 0
        while True:
            queue = mp.Queue()

            # Spawn worker process
            p = mp.Process(
                target=_search_worker,
                args=(
                    self.cfg,
                    self.function_set,
                    X,
                    y,
                    seed,
                    queue
                )
            )
            p.start()

            # Block main process for up to `timeout` seconds
            p.join(timeout=timeout)

            # Timeout check: kill process and restart loop
            if p.is_alive():
                print(f"Execution exceeded {timeout} seconds. Terminating process and restarting...")
                p.terminate()
                p.join()
                continue  # Restarts the search loop from the beginning

            # Successful completion check
            if not queue.empty():
                res = queue.get()
                tokens = res["tokens"]
                err = res["err"]
                expr = res["expr"]

                # Update main process state
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=err)
                self.errors.append(err)
                self.sympy.append(expr)
                break
            else:
                if ct == 0:
                    ct += 1
                    print("Worker process exited unexpectedly without returning data. Retrying...")
                else:
                    print("Iteration omitted")
                    tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
                    sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                    self.errors.append(1000)
                    self.sympy.append(sp.sympify('0'))
                    break

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
