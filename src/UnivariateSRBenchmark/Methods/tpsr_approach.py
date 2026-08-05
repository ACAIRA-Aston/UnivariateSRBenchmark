import gc
import UnivariateSRBenchmark.Methods.TPSR as TPSR
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from UnivariateSRBenchmark.Methods.E2E.model.sklearn_wrapper import SymbolicTransformerRegressor
from SRToolkit.utils.types import ModelResult
import sys
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.envs import build_env
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test
from UnivariateSRBenchmark.Methods.TPSR.dyna_gym.agents.uct import UCT
from UnivariateSRBenchmark.Methods.TPSR.dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from UnivariateSRBenchmark.Methods.TPSR.rl_env import RLEnv
from UnivariateSRBenchmark.Methods.TPSR.default_pi import E2EHeuristic, NesymresHeuristic
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.metrics import compute_metrics

import warnings
warnings.filterwarnings("ignore")


sys.modules["symbolicregression"] = TPSR
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  # This deals with PosixPath errors when loading the model

# Tuning Grid
#   lambda_reg: [0.01, 0.1, 1.0, 5.0]
#   width: [2, 3, 4]
#   rollout: [1, 3, 6, 9]
#   beams: [1, 3]


@dataclass
class Params:
    eval_mcts_in_domain: bool = True
    lam: float = 0.1
    horizon: int = 200
    width: int = 3
    num_beams: int = 1
    rollout: int = 3
    no_seq_cache: bool = False
    no_prefix_cache: bool = True
    max_input_points: int = 200
    eval_input_length_modulo: int = 50
    prediction_sigmas: str = "1,2,4,8,16"
    save_results: bool = True
    debug: bool = True
    is_slurm_job: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam_length_penalty: int = 1
    backbone_model: str = 'e2e'
    beam_size: int = 5


@dataclass
class TPSRConfig(ApproachConfig):
    name: str = "TPSR"
    lambda_reg: float = 0.1
    width: int = 3
    rollout: int = 3
    beams: int = 1



class TPSRApproach(SR_approach):
    def __init__(self, lambda_reg: float = 0.1,
                        width: int = 3,
                        rollout: int = 3,
                        beams: int = 1):
        self.cfg = TPSRConfig(lambda_reg=lambda_reg, width=width, rollout=rollout, beams=beams)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.params = Params()
        self.params.lam = self.cfg.lambda_reg
        self.params.width = self.cfg.width
        self.params.rollout = self.cfg.rollout
        self.params.beams = self.cfg.beams
        self.params.debug = True
        self.params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.equation_env = build_env(self.params)

    def prepare(self):
        pass

    def main_e2e(self, samples):
        params = self.params
        model = Transformer(params=params, samples=samples)
        model.to(params.device)
        generations_ref, gen_len_ref = respond_to_batch(model, max_target_length=200, top_p=1.0,
                                                        sample_temperature=None)
        sequence_ref = generations_ref[0][:gen_len_ref - 1].tolist()

        rl_env = RLEnv(
            params=params,
            samples=samples,
            equation_env=model.env,
            model=model)

        dp = E2EHeuristic(
            equation_env=model.env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty=params.beam_length_penalty)

        agent = UCT(
            action_space=[],  # this will not be used as we have a default policy
            gamma=1.,  # no discounting
            ucb_constant=1.,
            horizon=params.horizon,
            rollouts=params.rollout,
            dp=dp,
            width=params.width,
            reuse_tree=True)

        # agent.display()

        done = False
        s = rl_env.state
        ret_all = []
        for t in range(params.horizon):
            if len(s) >= params.horizon:
                print(f'Cannot process programs longer than {params.horizon}. Stop here.')
                break
            if done:
                break
            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)
            if t == 0:
                real_root = agent.root
            if params.debug:
                # print the current tree
                # print_tree(agent.root, equation_env.equation_id2word)
                ret = convert_to_json(agent.root, rl_env, act)
                ret_all.append(ret)

                # print('took action:')
                # print(repr(model.env.equation_id2word[act]))
                # print('========== state (excluding prompt) ==========')
                # print(s)

            update_root(agent, act, s)
            dp.update_cache(s)

        # y_mcts, mcts_str, mcts_tree = pred_for_sample_no_refine(model, equation_env, s, samples['x_to_fit'])
        y_mcts_refine, mcts_str_refine, mcts_tree_refine = refine_for_sample(params, model, model.env, s,
                                                                             samples['x_to_fit'], samples['y_to_fit'])

        return mcts_str_refine


    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y
        data = np.concatenate((X, y[:, None]), axis=1)
        samples = {'x_to_fit': 0, 'y_to_fit': 0, 'x_to_pred': 0, 'y_to_pred': 0}
        samples['x_to_fit'] = [data[:, :-1]]
        samples['y_to_fit'] = [data[:, -1].reshape(-1, 1)]
        samples['x_to_pred'] = [data[:, :-1]]
        samples['y_to_pred'] = [data[:, -1].reshape(-1, 1)]

        # equation_env = build_env(self.params)

        mcts_str_refine = self.main_e2e(samples=samples)
        model = sp.parse_expr(mcts_str_refine)

        if model is not None and not sr_evaluator.should_stop:
            X_syms = {f'x_0':sp.Symbol('x_0')}
            expr = model.subs(X_syms['x_0'], sp.Symbol(f'X_0'))
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
                self.errors.append(err)
                self.sympy.append(expr)
            except ValueError:
                print("Iteration omitted")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def from_config(self, config: dict) -> "TPSRApproach":
        return self(lambda_reg=config.get("lambda_reg", 0.1),
                    width=config.get("width", 3),
                    rollout=config.get("rollout", 3),
                    beams=config.get("beams", 1)
        )
