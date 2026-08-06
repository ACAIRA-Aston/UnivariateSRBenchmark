import os
import UnivariateSRBenchmark.Methods.nesymres as NeSymReS
import sys
import omegaconf
import numpy as np
from dataclasses import dataclass
from typing import Optional
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

from src.UnivariateSRBenchmark.Methods.nesymres.architectures.model import Model
from src.UnivariateSRBenchmark.Methods.nesymres.utils import load_metadata_hdf5
from src.UnivariateSRBenchmark.Methods.nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify
import json

import warnings
warnings.filterwarnings("ignore")


sys.modules["symbolicregression"] = NeSymReS
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  # This deals with PosixPath errors when loading the model

# Tuning Grid
#   beam_size: [2, 5, 10, 15]
#   bfgs_restarts: [2, 4, 8, 16]

# Important!: Install pytorch_lightning and hydra


@dataclass
class NeSymReSConfig(ApproachConfig):
    name: str = "NeSymReS"
    beam_size: int = 2
    bfgs_restarts: int = 2


class NeSymReSApproach(SR_approach):
    def __init__(self, beam_size: int = 2,
                       bfgs_restarts: int = 2):
        self.cfg = NeSymReSConfig(beam_size=beam_size,
                                  bfgs_restarts=bfgs_restarts)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        ## Load equation configuration and architecture configuration
        with open(Path(os.path.join(get_project_root(), 'UnivariateSRBenchmark\\Methods\\nesymres\\eq_setting.json')), 'r') as json_file:
            eq_setting = json.load(json_file)
        cfg = omegaconf.OmegaConf.load(Path(os.path.join(get_project_root(), 'UnivariateSRBenchmark\\Methods\\nesymres\\config.yaml')))

        ## Set up BFGS load rom the hydra config yaml
        bfgs = BFGSParams(
            activated=cfg.inference.bfgs.activated,
            n_restarts=self.cfg.bfgs_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

        params_fit = FitParams(word2id=eq_setting["word2id"],
                               id2word={int(k): v for k, v in eq_setting["id2word"].items()},
                               una_ops=eq_setting["una_ops"],
                               bin_ops=eq_setting["bin_ops"],
                               total_variables=list(eq_setting["total_variables"]),
                               total_coefficients=list(eq_setting["total_coefficients"]),
                               rewrite_functions=list(eq_setting["rewrite_functions"]),
                               bfgs=bfgs,
                               beam_size=self.cfg.beam_size)

        ## Load architecture, set into eval mode, and pass the config parameters
        weights_path = Path(os.path.join(get_project_root(), 'UnivariateSRBenchmark\\Methods\\nesymres\\100M.ckpt'))
        self.model = Model.load_from_checkpoint(checkpoint_path=weights_path, cfg=cfg.architecture, weights_only=False)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        print("Model successfully loaded on ", self.model.device)
        self.fitfunc = partial(self.model.fitfunc, cfg_params=params_fit)


    def prepare(self):
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        y = sr_evaluator.y
        X = np.zeros((len(sr_evaluator.X), 3))  # Nesymres always receives input matrices of this shape
        X[:, 0] = sr_evaluator.X[:, 0]
        X = torch.from_numpy(X).cuda()
        y = torch.from_numpy(y).cuda()

        model = self.fitfunc(X, y)

        # Extract and evaluate expressions
        program = model['best_bfgs_preds'][0]

        if program is not None and not sr_evaluator.should_stop:
            X_syms = {f'x_1':sp.Symbol('x_1')}
            expr = sp.sympify(program).subs(X_syms['x_1'], sp.Symbol(f'X_0'))
            print("Retrieved expression: ", expr)

            if 'X_0' not in str(expr.free_symbols):
                if expr.is_real:
                    values = np.repeat(float(expr), len(X), axis=0)[:, None]
                else:
                    values = np.repeat(1000, len(X), axis=0)[:, None]  # Penalize these cases
            else:
                f = sp.lambdify(sp.Symbol('X_0'), expr, modules="numpy")
                values = f(sr_evaluator.X)
            try:
                tokens, const_values = sympy_to_srtoolkit_tokens(expr)
                err = np.sqrt(np.mean((values[:, 0] - sr_evaluator.y) ** 2))
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
    def from_config(self, config: dict) -> "NeSymReSApproach":
        return self(beam_size=config.get("beam_size", 2),
                    bfgs_restarts=config.get("bfgs_restarts", 2)
        )
