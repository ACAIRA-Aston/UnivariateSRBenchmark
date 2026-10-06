import numpy as np
from dataclasses import dataclass
from typing import Optional
from feat import FeatRegressor
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from UnivariateSRBenchmark.utils import *

import warnings
warnings.filterwarnings("ignore")

# Tuning Grid
# pop_size: [100, 250, 500, 1000]
# gens: [50, 100, 250, 500]
# archive_size: [10, 20, 50, 100]
# complexity_weight: [0.001, 0.01, 0.1, 0.5]
# mutation_rate: [0.05, 0.1, 0.15, 0.2]

# NOTE on library mapping (verified against cavalab/feat's feat/feat.py + src/feat.cc):
# - pop_size, gens, max_depth, max_dim map 1:1 onto Feat's constructor.
# - Feat only exposes a single `cross_rate` (rate of crossover vs. mutation);
#   there is no separate `mutation_rate` knob, so mutation_rate is applied as
#   `cross_rate = 1 - mutation_rate` here.
# - Feat has no literal `archive_size` or `complexity_weight` constructor args.
#   Feat instead keeps a Pareto archive trading off `fitness` vs `complexity`
#   (multi-objective NSGA-II over objectives=["fitness","complexity"]). We use
#   `archive_size` to cap how many Pareto-front individuals we consider, and
#   `complexity_weight` to pick among them via error + complexity_weight*complexity,
#   which is the natural way to turn a scalarizing weight into a selection rule
#   over a Pareto front. max_dim=1 keeps each individual a single symbolic
#   expression rather than a multi-feature linear ensemble, which is what makes
#   a single closed-form sympy expression meaningful here.


@dataclass
class FeatConfig(ApproachConfig):
    name: str = "Feat"
    pop_size: int = 250
    gens: int = 100
    archive_size: int = 20
    complexity_weight: float = 0.01
    mutation_rate: float = 0.1


class FeatApproach(SR_approach):
    def __init__(self, pop_size=250, gens=100, archive_size=20, complexity_weight=0.01, mutation_rate=0.1):
        self.cfg = FeatConfig(pop_size=pop_size,
                               gens=gens,
                               archive_size=archive_size,
                               complexity_weight=complexity_weight,
                               mutation_rate=mutation_rate)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.functions = "+,-,*,/,sin,cos,exp,log,sqrt,^2,^3"

    def prepare(self):
        pass

    def _feat_term_to_sympy(self, term: str) -> sp.Expr:
        """Converts a single FEAT feature/weight string into a sympy-parsable one.

        FEAT's printed features are already infix arithmetic (e.g.
        "sin(0.5023*x_0)"), but use two non-python-syntax conventions:
          - power is written as (W*EXPR)^(W*EXPR) -> **
          - protected sqrt is written as sqrt(|EXPR|) -> sqrt(Abs(EXPR))
        Variables are renamed from x_0 -> X_0 to match the toolkit convention.
        """
        s = term
        s = s.replace('sqrt(|', 'sqrt(Abs(').replace('|)', '))')
        s = s.replace('^', '**')
        s = s.replace('x_', 'X_')
        return sp.sympify(s)

    def _select_from_archive(self, model: FeatRegressor) -> Optional[str]:
        """Picks a model from FEAT's Pareto archive using complexity_weight,
        falling back to the single best model FEAT already selected if the
        archive isn't reachable for any reason."""
        try:
            archive = model.cfeat_.get_archive(True)  # True = Pareto front only
            archive = sorted(archive, key=lambda ind: ind.get('complexity', 0))[: self.cfg.archive_size]
            if archive:
                best = min(archive, key=lambda ind: ind['fitness'] + self.cfg.complexity_weight * ind['complexity'])
                return best['eqn']
        except Exception:
            pass
        return None

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        inds =np.arange(len(sr_evaluator.X))
        np.random.shuffle(inds)
        X = sr_evaluator.X[inds]
        y = sr_evaluator.y[inds]
        X2d = X.reshape(-1, 1) if X.ndim == 1 else X

        model = FeatRegressor(
            pop_size=self.cfg.pop_size,
            gens=self.cfg.gens,
            cross_rate=1.0 - self.cfg.mutation_rate,
            functions=self.functions,
            max_depth=3,
            max_dim=1,  # keep individuals single-expression, not multi-feature ensembles
            objectives=["fitness", "complexity"],
            random_state=seed if seed is not None else 0,
            verbosity=0,
        )
        model.fit(X2d, y)

        if sr_evaluator.should_stop:
            return

        eqn = self._select_from_archive(model)
        if eqn is None:
            # fall back to FEAT's own weight/feature table for its selected best model
            table = model.get_model(sort=True)
            terms = []
            offset = 0.0
            for line in table.splitlines()[1:]:
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                w, feat = parts
                if feat == 'offset':
                    offset = float(w)
                else:
                    terms.append(f"({w})*({feat})")
            eqn = " + ".join(terms) if terms else "0"
            eqn = f"({eqn}) + ({offset})"

        try:
            expr = self._feat_term_to_sympy(eqn)
            expr = sp.nsimplify(expr, rational=False)
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
    def from_config(self, config: dict) -> "FeatApproach":
        return self(
            pop_size=config.get("pop_size", 250),
            gens=config.get("gens", 100),
            archive_size=config.get("archive_size", 20),
            complexity_weight=config.get("complexity_weight", 0.01),
            mutation_rate=config.get("mutation_rate", 0.1),
        )
