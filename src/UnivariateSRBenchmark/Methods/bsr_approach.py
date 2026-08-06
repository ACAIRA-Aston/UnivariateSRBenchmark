import numpy as np
from dataclasses import dataclass
from typing import Optional
from autora.theorist.bsr import BSRRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
#   tree_num: [2, 4, 8]
#   iter_num: [20, 50, 100, 200]
#   beta: [-0.1, -1, -2, -5]


@dataclass
class BSRConfig(ApproachConfig):
    name: str = "BSR"
    tree_num: int = 2
    iter_num: int = 50
    beta: float = -1.0


class BSRApproach(SR_approach):
    def __init__(self, tree_num: int = 2,
                       iter_num: int = 50,
                       beta: float = -1.0):
        self.cfg = BSRConfig(tree_num=tree_num, iter_num=iter_num, beta=beta)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []


    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        model = BSRRegressor(itr_num = self.cfg.iter_num, beta = self.cfg.beta,
                             tree_num=self.cfg.tree_num)
        model.fit(X, y)

        # Extract and evaluate expressions
        print(model._model())
        program = (model._model()[0]).replace('x[0]', 'X_0')
        program = program.replace('[', '(').replace(']', ')')
        try:
            if not sr_evaluator.should_stop:
                expr = sp.sympify(program)

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
        except Exception:
            print("Iteration omitted")
            tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
            sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
            self.errors.append(1000)
            self.sympy.append(sp.sympify('0'))

        del model

    @classmethod
    def from_config(self, config: dict) -> "BSRApproach":
        return self(
            tree_num=config.get("tree_num", 2),
            iter_num=config.get("iter_num", 50),
            beta=config.get("beta", -1.0),
        )

    #####################################################################
    # IMPORTANT Change package site-packages\autora\theorist\bsr\funcs.py
    #####################################################################
    # Line 187: error = np.sum(np.square(y - output))

    # Line 808: for i in range(len(all_nodes)):

    # Lines 922-923:
    # try:
    #     alpha = min(min(log_r), 0)
    # except TypeError:
    #     alpha = min(log_r, 0)
    # test = np.random.uniform(0, 1)


    #########################################################################
    # IMPORTANT Change package site-packages\autora\theorist\bsr\operation.py
    #########################################################################
    # --- Added Unary Operators ---

    # def tan_op(operand: np.ndarray) -> np.ndarray:
    #     return np.tan(operand)
    #
    # def log_op(operand: np.ndarray, **params: Dict) -> np.ndarray:
    #     cutoff = params.get("cutoff", 1e-10)
    #     return np.log(np.abs(operand) + cutoff)
    #
    # def sqrt_op(operand: np.ndarray) -> np.ndarray:
    #     return np.sqrt(np.abs(operand))
    #
    # def abs_op(operand: np.ndarray) -> np.ndarray:
    #     return np.abs(operand)
    #
    # def asin_op(operand: np.ndarray) -> np.ndarray:
    #     # Clip input to [-1, 1] to avoid NaNs
    #     return np.arcsin(np.clip(operand, -1.0, 1.0))
    #
    # def atan_op(operand: np.ndarray) -> np.ndarray:
    #     return np.arctan(operand)
    #
    # def sinh_op(operand: np.ndarray) -> np.ndarray:
    #     # Clip to prevent extreme exponential overflow
    #     return np.sinh(np.clip(operand, -50.0, 50.0))
    #
    # def cosh_op(operand: np.ndarray) -> np.ndarray:
    #     # Clip to prevent extreme exponential overflow
    #     return np.cosh(np.clip(operand, -50.0, 50.0))
    #
    # def tanh_op(operand: np.ndarray) -> np.ndarray:
    #     return np.tanh(operand)
    #
    # def asinh_op(operand: np.ndarray) -> np.ndarray:
    #     return np.arcsinh(operand)
    #
    # def acosh_op(operand: np.ndarray, **params: Dict) -> np.ndarray:
    #     cutoff = params.get("cutoff", 1e-10)
    #     # Ensure operand >= 1.0
    #     return np.arccosh(1.0 + np.abs(operand - 1.0) + cutoff)
    #
    # def atanh_op(operand: np.ndarray) -> np.ndarray:
    #     # Clip input strictly inside (-1, 1) to avoid infinity/NaN
    #     return np.arctanh(np.clip(operand, -0.999999, 0.999999))


    #####################################################################
    # IMPORTANT Change package site-packages\autora\theorist\bsr\prior.py
    #####################################################################
