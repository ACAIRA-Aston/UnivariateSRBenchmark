import os
import UnivariateSRBenchmark.Methods.E2E as E2E
import sys
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from UnivariateSRBenchmark.Methods.E2E.model.sklearn_wrapper import SymbolicTransformerRegressor
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")


sys.modules["symbolicregression"] = E2E
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  # This deals with PosixPath errors when loading the model

# Tuning Grid
#   num_bags: [50, 100, 200]
#   beam_size: [1, 5, 10]
#   bfgs_restarts: [1, 5, 10]


@dataclass
class E2EConfig(ApproachConfig):
    name: str = "E2E"
    B: int = 50
    C: int = 1
    K: float = 1


class E2EApproach(SR_approach):
    def __init__(self, B: int = 50,
                       C: int = 1,
                       K: float = 1):
        self.cfg = E2EConfig(B=B,
                             C=C,
                             K=K)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        model_path = Path(os.path.join(get_project_root(), "UnivariateSRBenchmark\\Methods\\E2E\\model.pt"))
        self.model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
        print("Model successfully loaded on ", self.model.device)


    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        model = SymbolicTransformerRegressor(model=self.model,
                                             max_input_points=len(X)//self.cfg.B,
                                             n_trees_to_refine=self.cfg.K,
                                             rescale=True
                                             )
        model.fit(X, y)

        # Extract and evaluate expressions
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
        program = model.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
        for op, replace_op in replace_ops.items():
            program = program.replace(op, replace_op)

        if program is not None and not sr_evaluator.should_stop:
            X_syms = {f'x_0':sp.Symbol('x_0')}
            expr = sp.sympify(program).subs(X_syms['x_0'], sp.Symbol(f'X_0'))
            print("Retrieved expression: ", expr)

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
                # for key in sr_evaluator.models.keys():
                #     sr_evaluator.models[key].error = err

                self.errors.append(err)
                self.sympy.append(expr)
            except ValueError:
                print("Iteration omitted")

    @classmethod
    def from_config(self, config: dict) -> "E2EApproach":
        return self(B=config.get("B", 50),
                    C=config.get("C", 1),
                    K=config.get("K", 1),
        )
