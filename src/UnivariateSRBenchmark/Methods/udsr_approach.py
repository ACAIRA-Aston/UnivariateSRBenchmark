import collections
import collections.abc
collections.Mapping = collections.abc.Mapping  # THIS FIX SOME PROBLEMS RELATED WITH DSO'S VERSION

from dso import DeepSymbolicRegressor
import numpy as np
from dataclasses import dataclass
from typing import Optional
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
#   n_samples: [1000, 500000, 1000000]
#   lr: [0.0003, 0.0005, 0.001, 0.01]
#   risk: [0.05, 0.10, 0.15, 0.5]

from dso.library import Token
from dso.functions import function_map

# 1. Define custom function logic (using safe protected versions to prevent overflows)
def safe_sinh(x1):
    return np.sinh(np.clip(x1, -50, 50))

def safe_cosh(x1):
    return np.cosh(np.clip(x1, -50, 50))

def safe_tanh(val):
    return np.tanh(val)

# 2. Add Tokens directly to DSO's function_map
function_map["asin"] = Token(np.arcsin, "asin", arity=1, complexity=4)
function_map["atan"] = Token(np.arctan, "atan", arity=1, complexity=4)
function_map["sinh"] = Token(safe_sinh, "sinh", arity=1, complexity=4)
function_map["cosh"] = Token(safe_cosh, "cosh", arity=1, complexity=4)
function_map["tanh"] = Token(safe_tanh, "tanh", arity=1, complexity=4)
function_map["asinh"] = Token(np.arcsinh, "asinh", arity=1, complexity=4)
function_map["acosh"] = Token(np.arccosh, "acosh", arity=1, complexity=4)
function_map["atanh"] = Token(np.arctanh, "atanh", arity=1, complexity=4)


@dataclass
class UDSRConfig(ApproachConfig):
    name: str = "UDSR"
    max_length: int = 30
    n_samples: int = 1000
    lr: float = 0.8
    risk: float = 0.05


class UDSRApproach(SR_approach):
    def __init__(self, max_length: int = 30,
                       n_samples: int = 1000,
                       lr: float = 0.1,
                       risk: float = 0.05,):
        self.cfg = UDSRConfig(max_length=max_length,
                             n_samples=n_samples,
                             lr=lr,
                             risk=risk)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []


    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        inds =np.arange(len(sr_evaluator.X))
        np.random.shuffle(inds)
        X = sr_evaluator.X[inds]
        y = sr_evaluator.y[inds]
        
        UDSR_config = {
            "task": {
                "task_type": "regression",
                "function_set": ["add", "sub", "mul", "div",
                                 "sin", "tan", "asin", "atan",
                                 "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                                 "exp", "log", "sqrt", "abs"],
            },
            # Hyperparameters for generation and sampling
            "training": {
                         "n_samples": self.cfg.n_samples,
                         "epsilon": self.cfg.risk,},
            "policy_optimizer": {"learning_rate": self.cfg.lr,},
            # "policy": {"max_length": self.cfg.max_length}
            "poly_optimizer_params": {
                "degree": 3,
                "coef_tol": 1e-6,
                "regressor": "dso_least_squares"
            }
        }

        model = DeepSymbolicRegressor(UDSR_config)
        model.fit(X, y)

        # Extract and evaluate expressions
        program = model.program_.sympy_expr
        if program is not None and not sr_evaluator.should_stop:
            X_syms = {f'x1':sp.Symbol('x1')}
            expr = program.subs(X_syms['x1'], sp.Symbol(f'X_0'))

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
    def from_config(self, config: dict) -> "UDSRApproach":
        return self(
            max_length=config.get("max_length", 80),
            n_samples=config.get("n_samples", 1000),
            lr=config.get("lr", 0.001),
            risk=config.get("risk", 0.05),
        )
