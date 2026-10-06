import sys
import types
if "pkg_resources" not in sys.modules:
    dummy_pkg = types.ModuleType("pkg_resources")
    dummy_pkg.declare_namespace = lambda name: None
    sys.modules["pkg_resources"] = dummy_pkg
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pygpg.sk import GPGRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *
from SRToolkit.utils.types import ModelResult

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# population_size: [100, 250, 500, 1000]
# generations: [50, 100, 200, 500]
# max_tree_size: [15, 31, 63, 127]

# NOTE on library mapping (verified against marcovirgolin/gpg's src/globals.hpp
# CLI option table + src/pypkg/pygpg/sk.py): GPGRegressor(**kwargs) forwards
# every kwarg as a "-flag value" CLI option to the C++ backend. Population size
# is the "pop" flag, but it's only honored when the Interleaved Multistart
# Scheme (IMS) is disabled -- with IMS active (the default) pop_size is fixed
# internally to 64 regardless of "pop". We therefore always pass
# disable_ims=True so population_size takes effect. generations maps to "g".
# gpg constrains tree complexity by *depth* ("d"), not raw node count, so
# max_tree_size is converted to the depth of a full binary tree with that many
# nodes (max_tree_size = 2^(d+1) - 1) to get an equivalent bound.


@dataclass
class GPGomeaConfig(ApproachConfig):
    name: str = "GPGomea"
    population_size: int = 250
    generations: int = 100
    max_tree_size: int = 31


class GPGomeaApproach(SR_approach):
    def __init__(self, population_size=250, generations=100, max_tree_size=31):
        self.cfg = GPGomeaConfig(population_size=population_size,
                                  generations=generations,
                                  max_tree_size=max_tree_size)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.function_set = "+,-,*,/,sin,tan,exp,log,sqrt,asin,atan,sinh,cosh,tanh,asinh,acosh,atanh,abs"

    def prepare(self):
        pass

    def _max_tree_size_to_depth(self, max_tree_size: int) -> int:
        # inverse of max_tree_size = 2^(d+1) - 1 for a full binary tree of depth d
        d = round(math.log2(max_tree_size + 1) - 1)
        return max(1, d)

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        inds =np.arange(len(sr_evaluator.X))
        np.random.shuffle(inds)
        X = sr_evaluator.X[inds]
        y = sr_evaluator.y[inds]
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64).ravel()

        model = GPGRegressor(
            disable_ims=True,
            pop=self.cfg.population_size,
            g=self.cfg.generations,
            t=-1,
            e=-1,
            d=self._max_tree_size_to_depth(self.cfg.max_tree_size),
            fset=self.function_set,
            random_state=seed if seed is not None else -1,
            finetune=False,
            verbose=False,
        )
        try:
            model.fit(X, y)

            if not hasattr(model, "model") or model.model is None or sr_evaluator.should_stop:
                return

            try:
                # gpg's internal variable convention is x_0, x_1, ... ; model.model
                # is already a sympy expression, so we just rename the symbol.
                expr = model.model.subs(sp.Symbol('x_0'), sp.Symbol('X_0'))
            except Exception:
                print("Iteration omitted")
                return

            print('Expr', expr)
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
            except Exception as A:
                print("Iteration omitted")
                tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
                sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
                self.errors.append(1000)
                self.sympy.append(sp.sympify('0'))
        except Exception as A:
            print("Iteration omitted")
            tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
            sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
            self.errors.append(1000)
            self.sympy.append(sp.sympify('0'))


    @classmethod
    def from_config(self, config: dict) -> "GPGomeaApproach":
        return self(
            population_size=config.get("population_size", 250),
            generations=config.get("generations", 100),
            max_tree_size=config.get("max_tree_size", 31),
        )


    ################################
    # IMPORTANT
    ################################
    # In gpg/src/globals.hpp, modify Line 27 with this:
    # vector<Op*> all_operators = {
    #     new Add(), new Sub(), new Neg(), new Mul(), new Div(), new Inv(),
    #     new Sqrt(),
    #     new Sin(), new Cos(), new Tan(),
    #     new Exp(), new Log(),
    #     new Asin(), new Atan(),
    #     new Sinh(), new Cosh(), new Tanh(),
    #     new Asinh(), new Acosh(), new Atanh(),
    #     new Abs()
    #   };

    # Then, in operator.hpp, add these new operators:
    # // ---------------- NEW OPERATORS ----------------
    #
    # struct Tan : Fun {
    #   Op * clone() override { return new Tan(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "tan"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().tan().matrix(); }
    # };
    #
    # struct Exp : Fun {
    #   Op * clone() override { return new Exp(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "exp"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().exp().matrix(); }
    # };
    #
    # struct Asin : Fun {
    #   Op * clone() override { return new Asin(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "asin"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().asin().matrix(); }
    # };
    #
    # struct Atan : Fun {
    #   Op * clone() override { return new Atan(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "atan"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().atan().matrix(); }
    # };
    #
    # struct Sinh : Fun {
    #   Op * clone() override { return new Sinh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "sinh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().sinh().matrix(); }
    # };
    #
    # struct Cosh : Fun {
    #   Op * clone() override { return new Cosh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "cosh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().cosh().matrix(); }
    # };
    #
    # struct Tanh : Fun {
    #   Op * clone() override { return new Tanh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "tanh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().tanh().matrix(); }
    # };
    #
    # struct Asinh : Fun {
    #   Op * clone() override { return new Asinh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "asinh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().asinh().matrix(); }
    # };
    #
    # struct Acosh : Fun {
    #   Op * clone() override { return new Acosh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "acosh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().acosh().matrix(); }
    # };
    #
    # struct Atanh : Fun {
    #   Op * clone() override { return new Atanh(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "atanh"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().atanh().matrix(); }
    # };
    #
    # struct Abs : Fun {
    #   Op * clone() override { return new Abs(); }
    #   int arity() override { return 1; }
    #   string sym() override { return "abs"; }
    #   Vec apply(Mat & X) override { return X.col(0).array().abs().matrix(); }
    # };


    # Replace gpg/src/pypkg/pygpg/sk.py line 243 with:
    # f = sympy.lambdify(sympy.Symbol('x_0'), model, modules="numpy")