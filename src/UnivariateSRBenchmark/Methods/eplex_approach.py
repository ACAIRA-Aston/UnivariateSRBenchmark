import numpy as np
from dataclasses import dataclass
from typing import Optional
from ellyn import ellyn
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [100, 250, 500, 1000]
# generations: [50, 100, 200, 300]
# crossover_rate: [0.6, 0.7, 0.8, 0.9]
# mutation_rate: [0.05, 0.1, 0.15, 0.2]
# lexicase_epsilon: [0.01, 0.05, 0.1, 0.2]

# NOTE on library mapping: EPLEX here refers to cavalab/ellyn's epsilon-lexicase
# selection method (selection='epsilon_lexicase'). ellyn's constructor uses its
# own short names for these knobs: g=generations, popsize=population_size,
# rt_cross=crossover_rate, rt_mut=mutation_rate, lex_epsilon=lexicase_epsilon.
# These mappings were confirmed against the ellyn source (src/ellyn.py).


@dataclass
class EplexConfig(ApproachConfig):
    name: str = "Eplex"
    population_size: int = 250
    generations: int = 100
    tournament_size: int = 2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    lexicase_epsilon: float = 0.05


class EplexApproach(SR_approach):
    def __init__(self, population_size=250, generations=100, tournament_size=2, crossover_rate=0.7,
                 mutation_rate=0.1, lexicase_epsilon=0.05):
        self.cfg = EplexConfig(population_size=population_size,
                                generations=generations,
                                tournament_size=tournament_size,
                                crossover_rate=crossover_rate,
                                mutation_rate=mutation_rate,
                                lexicase_epsilon=lexicase_epsilon)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        # ellyn's function set is given as a flat list of single-character/-token
        # op codes; this mirrors the operators used in the gplearn approach as
        # closely as ellyn supports (see docs in ellyn.py: 'n','v','+','-','*','/',
        # 'sin','cos','exp','log','sqrt' among others).
        self.op_list = ['n', 'v', '+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt']
        # TODO: THE FUNCTION SET IS INCOMPLETE, IT SHOULD BE SOMETHING LIKE: self.op_list = [
        #             # Trigonometric
        #             "sin",
        #             "tan",
        #             "asin",
        #             "atan",
        #             # Hyperbolic
        #             "sinh",
        #             "cosh",
        #             "tanh",
        #             "asinh",
        #             "acosh",
        #             "atanh",
        #             # Exponential & Logarithmic
        #             "exp",
        #             "log",
        #             # Basic Arithmetic
        #             "sqrt",
        #             "abs",
        #             # Algebraic Structural Ops
        #             "square",
        #             "cube",
        #         ] ... adapt this set to this method


    def prepare(self):
        pass

    def _eplex_eqn_to_sympy(self, eqn_str: str) -> sp.Expr:
        """Converts an ellyn stack_2_eqn() string into a sympy expression.

        ellyn's equation strings are already infix (e.g. "((x_0+3)*sin(x_0))"),
        but use a couple of non-python-syntax conventions that need translating:
          - power is written as EXPR^N (e.g. "(x_0^2)")      -> EXPR**N
          - protected sqrt is written as sqrt(|EXPR|)         -> sqrt(Abs(EXPR))
          - inverse trig is written as arcsin/arccos          -> asin/acos
          - variables are written as x_0, x_1, ...            -> X_0, X_1, ...
        """
        s = eqn_str
        s = s.replace('sqrt(|', 'sqrt(Abs(').replace('|)', '))')
        s = s.replace('^', '**')
        s = s.replace('arcsin', 'asin').replace('arccos', 'acos')
        s = s.replace('x_', 'X_')
        return sp.sympify(s)

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y
        X2d = X.reshape(-1, 1) if X.ndim == 1 else X

        model = ellyn(
            g=self.cfg.generations,
            popsize=self.cfg.population_size,
            selection='epsilon_lexicase',
            tourn_size=self.cfg.tournament_size,
            rt_cross=self.cfg.crossover_rate,
            rt_mut=self.cfg.mutation_rate,
            rt_rep=0,
            lex_eps_error_mad=True,
            lex_epsilon=self.cfg.lexicase_epsilon,
            op_list=self.op_list,
            random_state=seed,
            classification=False,
            verbosity=0,
        )
        model.fit(X2d, y)

        if model.best_estimator_ is not None and not sr_evaluator.should_stop:
            try:
                raw_eqn = model.stack_2_eqn(model.best_estimator_)
                expr = self._eplex_eqn_to_sympy(raw_eqn)
            except Exception:
                print("Iteration omitted")
                return

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
                tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                self.errors.append(1000)
                self.sympy.append(sp.sympify('0'))

    @classmethod
    def from_config(self, config: dict) -> "EplexApproach":
        return self(
            population_size=config.get("population_size", 250),
            generations=config.get("generations", 100),
            tournament_size=config.get("tournament_size", 2),
            crossover_rate=config.get("crossover_rate", 0.7),
            mutation_rate=config.get("mutation_rate", 0.1),
            lexicase_epsilon=config.get("lexicase_epsilon", 0.05),
        )
