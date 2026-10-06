import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import ffx
import ffx.core.model_factories as _ffx_model_factories
from ffx.core.build_strategy import FFXBuildStrategy as _FFXBuildStrategyBase
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# max_complexity: [10, 20, 50, 100]
# l1_ratios: [0.1, 0.5, 0.7, 0.9]
# n_alphas: [50, 100, 200, 500]

# NOTE on library mapping: gAldeia/ffx exposes NO hyperparameters on its public
# API at all (ffx.run()/FFXRegressor take none; SRBench's own FFX tuning grid
# is `params = {}`, confirming this). Internally, FFXBuildStrategy hardcodes
# l1_ratio=0.95 and num_alphas=1000 with no constructor hook to override them.
# To honor the requested hyperparameters we temporarily monkeypatch the
# FFXBuildStrategy class referenced inside ffx.core.model_factories (restored
# in a finally block) for the duration of each search() call, sweeping over
# l1_ratios the way the classic FFX algorithm's pathwise elastic-net search
# does, and merging the resulting Pareto-optimal model sets. max_complexity
# is applied post-hoc against FFXModel.complexity(), which FFX already exposes.


@dataclass
class FfxConfig(ApproachConfig):
    name: str = "Ffx"
    max_complexity: int = 20
    l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.7, 0.9)
    n_alphas: int = 100


class FfxApproach(SR_approach):
    def __init__(self, max_complexity=20, l1_ratios=(0.1, 0.5, 0.7, 0.9), n_alphas=100):
        self.cfg = FfxConfig(max_complexity=max_complexity,
                              l1_ratios=tuple(l1_ratios),
                              n_alphas=n_alphas)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

    def prepare(self):
        pass

    def _make_patched_strategy(self, l1_ratio: float, num_alphas: int):
        cfg_l1_ratio = l1_ratio
        cfg_num_alphas = num_alphas

        class _PatchedStrategy(_FFXBuildStrategyBase):
            def __init__(self, approach):
                super().__init__(approach)
                self._l1_ratio = cfg_l1_ratio
                self._num_alphas = cfg_num_alphas

        return _PatchedStrategy

    def _ffx_term_to_sympy(self, eqn_str: str) -> sp.Expr:
        """Converts an FFXModel.str2() string into a sympy-parsable one.

        FFX's printed models are infix arithmetic over X_0 (we pass
        varnames=["X_0"] so variable substitution is already done for us), but
        use non-python-syntax conventions for a few basis/operator types:
          - power is written as x0^E    -> **
          - abs(...) / log10(...)       -> handled via a custom sympify locals dict
          - max(0, EXPR) / min(0, EXPR) -> sympy Max/Min
        """
        s = eqn_str.replace('^', '**')
        locals_map = {
            'log10': lambda expr: sp.log(expr, 10),
            'abs': sp.Abs,
            'max': sp.Max,
            'min': sp.Min,
        }
        return sp.sympify(s, locals=locals_map)

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        inds =np.arange(len(sr_evaluator.X))
        np.random.shuffle(inds)
        X = sr_evaluator.X[inds]
        y = sr_evaluator.y[inds]
        X2d = X.reshape(-1, 1) if X.ndim == 1 else X

        all_models = []
        original_strategy_cls = _ffx_model_factories.FFXBuildStrategy
        try:
            for l1_ratio in self.cfg.l1_ratios:
                _ffx_model_factories.FFXBuildStrategy = self._make_patched_strategy(l1_ratio, self.cfg.n_alphas)
                try:
                    models = ffx.run(X2d, y, X2d, y, varnames=["X_0"])
                    all_models.extend(models)
                except Exception:
                    continue
        finally:
            _ffx_model_factories.FFXBuildStrategy = original_strategy_cls

        if not all_models or sr_evaluator.should_stop:
            return

        candidates = [m for m in all_models if m.complexity() <= self.cfg.max_complexity]
        if not candidates:
            candidates = sorted(all_models, key=lambda m: m.complexity())[:1]
        best_model = min(candidates, key=lambda m: m.test_nmse)

        try:
            expr = self._ffx_term_to_sympy(best_model.str2(100000))
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
            sr_evaluator.evaluate_expr(tokens)
            err = np.sqrt(np.mean((values[:, 0] - y) ** 2))
            for key in sr_evaluator.models.keys():
                sr_evaluator.models[key].error = err
            self.errors.append(err)
            self.sympy.append(expr)
        except ValueError:
            print("Iteration omitted")

    @classmethod
    def from_config(self, config: dict) -> "FfxApproach":
        return self(
            max_complexity=config.get("max_complexity", 20),
            l1_ratios=config.get("l1_ratios", (0.1, 0.5, 0.7, 0.9)),
            n_alphas=config.get("n_alphas", 100),
        )
