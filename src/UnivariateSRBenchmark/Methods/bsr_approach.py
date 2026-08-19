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
        try:
            model.fit(X, y)

            # Extract and evaluate expressions
            print(model._model())
            program = (model._model()[0]).replace('x[0]', 'X_0')
            program = program.replace('[', '(').replace(']', ')')
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
    # from typing import Callable, Dict, Union
    #
    # import numpy as np
    # from scipy.stats import norm
    #
    # from .misc import normalize_prior_dict
    # from .operation import (
    #     abs_op,
    #     acosh_op,
    #     asin_op,
    #     asinh_op,
    #     atan_op,
    #     atanh_op,
    #     cos_op,
    #     cosh_op,
    #     div_op,
    #     exp_op,
    #     inv_op,
    #     linear_op,
    #     log_op,
    #     make_pow_op,
    #     minus_op,
    #     multiply_op,
    #     neg_op,
    #     plus_op,
    #     sin_op,
    #     sinh_op,
    #     sqrt_op,
    #     tan_op,
    #     tanh_op,
    # )
    #
    # def _get_ops_with_arity():
    #     """
    #     Get the operator function and arity (number of operands) of each operator.
    #
    #     Returns:
    #         ops_fn_and_arity: a dictionary that maps operator name to a list, where
    #             the first item is the operator function and the second is the number of
    #             operands that it takes.
    #     """
    #     ops_fn_and_arity = {
    #         # Linear & Power
    #         "ln": [linear_op, 1],
    #         "pow2": [make_pow_op(2), 1],
    #         "pow3": [make_pow_op(3), 1],
    #         # Basic Unary
    #         "exp": [exp_op, 1],
    #         "inv": [inv_op, 1],
    #         "neg": [neg_op, 1],
    #         "abs": [abs_op, 1],
    #         "log": [log_op, 1],
    #         "sqrt": [sqrt_op, 1],
    #         # Trigonometric Unary
    #         "sin": [sin_op, 1],
    #         "cos": [cos_op, 1],
    #         "tan": [tan_op, 1],
    #         "asin": [asin_op, 1],
    #         "atan": [atan_op, 1],
    #         # Hyperbolic Unary
    #         "sinh": [sinh_op, 1],
    #         "cosh": [cosh_op, 1],
    #         "tanh": [tanh_op, 1],
    #         "asinh": [asinh_op, 1],
    #         "acosh": [acosh_op, 1],
    #         "atanh": [atanh_op, 1],
    #         # Binary
    #         "+": [plus_op, 2],
    #         "*": [multiply_op, 2],
    #         "-": [minus_op, 2],
    #         "/": [div_op, 2],
    #     }
    #     return ops_fn_and_arity
    #
    # def linear_init(**hyper_params) -> Dict:
    #     """
    #     Initialization function for the linear operator. Two parameters, slope
    #     (a) and intercept (b) are initialized.
    #
    #     Arguments:
    #         hyper_params: the dictionary for hyperparameters. Specifically, this
    #             function requires `sigma_a` and `sigma_b` to be present.
    #     Returns:
    #         a dictionary with initialized `a` and `b` parameters.
    #     """
    #     sigma_a, sigma_b = hyper_params.get("sigma_a", 1), hyper_params.get("sigma_b", 1)
    #     return {
    #         "a": norm.rvs(loc=1, scale=np.sqrt(sigma_a)),
    #         "b": norm.rvs(loc=0, scale=np.sqrt(sigma_b)),
    #     }
    #
    # def _get_ops_init() -> Dict[str, Union[Callable, object]]:
    #     """
    #     Get the initialization functions for operators that require additional
    #     parameters.
    #
    #     Returns:
    #         ops_init: a dictionary that maps operator name to either a parameter
    #             dict (in the case that the initialization is hard-coded) or an
    #             initialization function (when it is randomized). The dictionary
    #             value will be used in growing the `node` (see `funcs_legacy.py`).
    #     """
    #     ops_init = {
    #         "ln": linear_init,
    #         "inv": {"cutoff": 1e-10},
    #         "exp": {"cutoff": 1e-10},
    #         "log": {"cutoff": 1e-10},
    #         "acosh": {"cutoff": 1e-10},
    #         "/": {"cutoff": 1e-10},
    #     }
    #     return ops_init
    #
    # def _get_prior(prior_name: str, prob: bool = True) -> Dict[str, float]:
    #     prior_dict = {
    #         "Uniform": {
    #             "neg": 1.0,
    #             "sin": 1.0,
    #             "cos": 1.0,
    #             "tan": 1.0,
    #             "asin": 1.0,
    #             "atan": 1.0,
    #             "sinh": 1.0,
    #             "cosh": 1.0,
    #             "tanh": 1.0,
    #             "asinh": 1.0,
    #             "acosh": 1.0,
    #             "atanh": 1.0,
    #             "pow2": 1.0,
    #             "pow3": 1.0,
    #             "exp": 1.0,
    #             "log": 1.0,
    #             "sqrt": 1.0,
    #             "abs": 1.0,
    #             "+": 1.0,
    #             "*": 1.0,
    #             "-": 1.0,
    #             "/": 1.0,
    #             "inv": 1.0,
    #             "ln": 1.0,
    #         },
    #         "Guimera2020": {
    #             "neg": 3.350846072163632,
    #             "sin": 5.965917796154835,
    #             "cos": 5.452564657261127,
    #             "tan": 5.0,
    #             "asin": 1.0,
    #             "atan": 1.0,
    #             "sinh": 1.0,
    #             "cosh": 1.0,
    #             "tanh": 1.0,
    #             "asinh": 1.0,
    #             "acosh": 1.0,
    #             "atanh": 1.0,
    #             "pow2": 3.3017352779079734,
    #             "pow3": 5.9907496760026175,
    #             "exp": 4.768665265735502,
    #             "log": 4.0,
    #             "sqrt": 3.0,
    #             "abs": 1.0,
    #             "+": 5.808163661224514,
    #             "*": 5.002213595420244,
    #             "-": 1.0,
    #             "/": 5.0,
    #             "inv": 1.0,
    #             "ln": 1.0,
    #         },
    #     }
    #     assert prior_dict[prior_name] is not None, "prior key not recognized"
    #     if prob:
    #         normalize_prior_dict(prior_dict[prior_name])
    #     return prior_dict[prior_name]
    #
    # def get_prior_dict(prior_name="Uniform"):
    #     """
    #     Get the dictionary of prior information as well as several list of key operator properties
    #
    #     Argument:
    #         prior_name: the name of the prior dictionary to use
    #
    #     Returns:
    #         ops_name_lst: the list of operator names
    #         ops_weight_lst: the list of operator weights
    #         prior_dict: the dictionary of operator prior information
    #     """
    #     ops_prior = _get_prior(prior_name)
    #     ops_init = _get_ops_init()
    #     ops_fn_and_arity = _get_ops_with_arity()
    #
    #     ops_name_lst = list(ops_prior.keys())
    #     ops_weight_lst = list(ops_prior.values())
    #     prior_dict = {
    #         k: {
    #             "init": ops_init.get(k, {}),
    #             "fn": ops_fn_and_arity[k][0],
    #             "arity": ops_fn_and_arity[k][1],
    #             "weight": ops_prior[k],
    #         }
    #         for k in ops_prior
    #     }
    #
    #     return ops_name_lst, ops_weight_lst, prior_dict
    #
    # def get_prior_list(prior_name="Uniform"):
    #     """
    #     Get a dictionary of key prior properties
    #
    #     Argument:
    #         prior_name: the name of the prior dictionary to use
    #
    #     Returns:
    #         a dictionary that maps a prior property (e.g. `name`) to the list of such properties
    #             for each operator.
    #     """
    #     ops_prior = _get_prior(prior_name)
    #     ops_init = _get_ops_init()
    #     ops_fn_and_arity = _get_ops_with_arity()
    #
    #     ops_name_lst = list(ops_prior.keys())
    #     ops_weight_lst = list(ops_prior.values())
    #     ops_init_lst = [ops_init.get(k, None) for k in ops_name_lst]
    #     ops_fn_lst = [ops_fn_and_arity[k][0] for k in ops_name_lst]
    #     ops_arity_lst = [ops_fn_and_arity[k][1] for k in ops_name_lst]
    #     return {
    #         "name": ops_name_lst,
    #         "weight": ops_weight_lst,
    #         "init": ops_init_lst,
    #         "fn": ops_fn_lst,
    #         "arity": ops_arity_lst,
    #     }

