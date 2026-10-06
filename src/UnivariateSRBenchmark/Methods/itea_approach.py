from dataclasses import dataclass
from typing import Optional
import warnings

from itea.regression import ITEA_regressor
import numpy as np
import sympy as sp
from SRToolkit.approaches.sr_approach import ApproachConfig, SR_approach
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils.types import ModelResult
from UnivariateSRBenchmark.utils import *

warnings.filterwarnings("ignore")


@dataclass
class IteaConfig(ApproachConfig):
    name: str = "Itea"
    pop_size: int = 250
    gens: int = 100
    max_terms: int = 10
    max_exponent: int = 2
    mutation_rate: float = 0.1


class IteaApproach(SR_approach):

    def __init__(
        self,
        pop_size=250,
        gens=100,
        max_terms=10,
        max_exponent=2,
        mutation_rate=0.1,
    ):
        self.cfg = IteaConfig(
            pop_size=pop_size,
            gens=gens,
            max_terms=max_terms,
            max_exponent=max_exponent,
            mutation_rate=mutation_rate,
        )
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        # --- Protected Safe Functions ---
        def _protected_exp(x):
            return np.exp(np.clip(x, -100, 100))

        def _protected_log(x):
            return np.log(np.abs(x) + 1e-8)

        def _protected_sqrt(x):
            return np.sqrt(np.abs(x))

        def _cube(x):
            return np.power(np.clip(x, -1e100, 1e100), 3)

        def _protected_asin(x):
            return np.arcsin(np.clip(x, -1.0, 1.0))

        def _protected_sinh(x):
            return np.sinh(np.clip(x, -50.0, 50.0))

        def _protected_cosh(x):
            return np.cosh(np.clip(x, -50.0, 50.0))

        def _protected_acosh(x):
            return np.arccosh(1.0 + np.abs(x - 1.0) + 1e-8)

        def _protected_atanh(x):
            return np.arctanh(np.clip(x, -0.999999, 0.999999))

        # --- Explicit Analytical Derivatives (tfuncs_dx) ---
        self.tfuncs_dx = {
            "id": lambda x: np.ones_like(x),
            "sin": np.cos,
            "cos": lambda x: -np.sin(x),
            "tan": lambda x: 1.0 / (np.cos(x) ** 2 + 1e-8),
            "exp": _protected_exp,
            "log": lambda x: np.sign(x) / (np.abs(x) + 1e-8),
            "sqrt": lambda x: np.sign(x) / (2 * np.sqrt(np.abs(x)) + 1e-8),
            "abs": lambda x: np.sign(x),
            "cube": lambda x: 3 * np.clip(x, -1e100, 1e100) ** 2,
            "asin": lambda x: 1.0 / np.sqrt(1.0 - np.clip(x, -0.999999, 0.999999) ** 2),
            "atan": lambda x: 1.0 / (1.0 + x**2),
            "sinh": _protected_cosh,
            "cosh": _protected_sinh,
            "tanh": lambda x: 1.0 - np.tanh(x) ** 2,
            "asinh": lambda x: 1.0 / np.sqrt(x**2 + 1.0),
            "acosh": lambda x: 1.0 / np.sqrt((1.0 + np.abs(x - 1.0) + 1e-8) ** 2 - 1.0),
            "atanh": lambda x: 1.0 / (1.0 - np.clip(x, -0.999999, 0.999999) ** 2),
        }

        # --- Unary Transformation Functions (tfuncs) ---
        self.tfuncs = {
            "id": lambda x: x,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": _protected_exp,
            "log": _protected_log,
            "sqrt": _protected_sqrt,
            "abs": np.abs,
            "cube": _cube,
            "asin": _protected_asin,
            "atan": np.arctan,
            "sinh": _protected_sinh,
            "cosh": _protected_cosh,
            "tanh": np.tanh,
            "asinh": np.arcsinh,
            "acosh": _protected_acosh,
            "atanh": _protected_atanh,
        }

        # --- SymPy Converters ---
        self.converter = {
            # Unbounded / Linear / Polynomial
            "id": lambda x: x,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": lambda x: sp.exp(sp.Piecewise((100, x > 100), (-100, x < -100), (x, True))),
            "cube": lambda x: x ** 3,
            "abs": sp.Abs,

            # Protected Positive / Logarithmic / Root Functions
            "log": lambda x: sp.log(sp.Abs(x) + 1e-8),
            "sqrt": lambda x: sp.sqrt(sp.Abs(x)),

            # Protected Inverse Trigonometric Functions
            "asin": lambda x: sp.asin(sp.Piecewise((1.0, x > 1.0), (-1.0, x < -1.0), (x, True))),
            "atan": sp.atan,

            # Protected Hyperbolic Functions
            "sinh": lambda x: sp.sinh(sp.Piecewise((50.0, x > 50.0), (-50.0, x < -50.0), (x, True))),
            "cosh": lambda x: sp.cosh(sp.Piecewise((50.0, x > 50.0), (-50.0, x < -50.0), (x, True))),
            "tanh": sp.tanh,

            # Protected Inverse Hyperbolic Functions
            "asinh": sp.asinh,
            "acosh": lambda x: sp.acosh(1.0 + sp.Abs(x - 1.0) + 1e-8),
            "atanh": lambda x: sp.atanh(
                sp.Piecewise((0.999999, x > 0.999999), (-0.999999, x < -0.999999), (x, True))
            ),
        }

    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        inds =np.arange(len(sr_evaluator.X))
        np.random.shuffle(inds)
        X = sr_evaluator.X[inds]
        y = sr_evaluator.y[inds]

        model = ITEA_regressor(
            popsize=self.cfg.pop_size,
            gens=self.cfg.gens,
            max_terms=self.cfg.max_terms,
            expolim=(-self.cfg.max_exponent, self.cfg.max_exponent),
            tfuncs=self.tfuncs,
            tfuncs_dx=self.tfuncs_dx,
            labels=["X_0"],
            random_state=seed,
            verbose=10,
        )
        model.fit(X, y)

        if not sr_evaluator.should_stop:
            expr_str = model.bestsol_.to_str(places=15)
            expr_str = expr_str.replace("^", "**")
            X_syms = {"X_0": sp.Symbol("X_0", real=True)}
            expr = sp.sympify(expr_str, locals={**self.converter, **X_syms})

            if "X_0" not in str(expr.free_symbols):
                if expr.is_real:
                    values = np.repeat(float(expr), len(X), axis=0)[:, None]
                else:
                    values = np.repeat(
                        1000, len(X), axis=0
                    )[:, None]  # Penalize non-real constants
            else:
                f = sp.lambdify(sp.Symbol("X_0"), expr, modules="numpy")
                values = f(X)
            try:
                try:
                    tokens, const_values = sympy_to_srtoolkit_tokens(expr)
                except ValueError:
                    print("Not tokenizable (yet), using a placeholder instead")
                    tokens = str(expr)
                err = np.sqrt(np.mean((values[:, 0] - y) ** 2))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=err)
                self.errors.append(err)
                self.sympy.append(expr)
            except ValueError:
                print("Iteration omitted")
                tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify("0"))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                self.errors.append(1000)
                self.sympy.append(sp.sympify("0"))

    @classmethod
    def from_config(cls, config: dict) -> "IteaApproach":
        return cls(
            pop_size=config.get("pop_size", 250),
            gens=config.get("gens", 100),
            max_terms=config.get("max_terms", 10),
            max_exponent=config.get("max_exponent", 2),
            mutation_rate=config.get("mutation_rate", 0.1),
        )

    #############################################################################
    # IMPORTANT Change package site-packages\itea\regression\_ITExpr_regressor.py
    #############################################################################
    # Line 185--188:
    # if self.fitness_f == 'rmse' or self.fitness_f == None:
    #     self._fitness = np.sqrt(mean_squared_error(pred, y))
    # elif self.fitness_f == 'mse':
    #     self._fitness = mean_squared_error(pred, y)

    #############################################################################
    # IMPORTANT Change package site-packages\itea\regression\_ITEA_regressor.py
    #############################################################################
    # Comment or remove Line 209: # self._explain_bestsol(self.bestsol_, X, y) (NOT NEEDED FOR UNIVARIATE)
