import os
import re
import glob
import shutil
import tempfile
import subprocess
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
# population_size: [200, 500, 1000, 2000]
# generations: [50, 100, 250, 500]
# crossover_rate: [0.5, 0.7, 0.8, 0.9]
# mutation_rate: [0.05, 0.1, 0.15, 0.2]
# lasso_lambda: [0.001, 0.01, 0.1, 0.5]

# =============================================================================
# IMPORTANT - please read before using this file
# =============================================================================
# Unlike the other four approaches, MRGP (Arnaldo, Krawiec & O'Reilly, 2014,
# https://flexgp.github.io/gp-learners/mrgp.html) has no Python package. It is
# distributed only as a compiled Java tool (mrgp.jar) with a documented but
# non-exhaustive CLI:
#     java -jar mrgp.jar -train <csv> -minutes <N> -properties <props_file>
# The FlexGP tutorial page confirms pop_size, terminal_set, function_set,
# tourney_size and mutation_rate as properties-file keys, and confirms the run
# writes pareto.txt / leastComplex.txt / mostAccurate.txt / knee.txt /
# bestModelGeneration.txt. It does NOT document property names for
# generations, crossover_rate, or a LASSO regularization strength, and the
# source is not publicly available to verify them against (mrgp.jar is a
# closed build; the historical Python wrapper referenced in cavalab/srbench,
# experiment/methods/src/mrgp, has since been removed from that repo).
#
# This file therefore takes a best-effort approach: it wraps the jar via
# subprocess, passes the confirmed properties, and ALSO passes generations /
# crossover_rate / lambda under plausible property names (commented below) as
# a best guess, backed by a hard "-minutes" wall-clock cap so a run can never
# hang indefinitely even if those guessed keys are ignored by the jar. If you
# have the actual mrgp.jar, please verify these property names (e.g. via
# `unzip -p mrgp.jar ... | strings` or any bundled docs) and adjust
# PROPERTY_NAME_OVERRIDES below accordingly.
#
# Requires: a working `java` on PATH, and MRGP_JAR_PATH pointing at a
# downloaded copy of https://flexgp.github.io/gp-learners/downloads/mrgp.jar
# =============================================================================

MRGP_JAR_PATH = os.environ.get("MRGP_JAR_PATH", "./mrgp.jar")

# Best-effort / unverified properties-file keys for the knobs the documented
# tutorial doesn't cover. Edit these if you confirm the real names.
PROPERTY_NAME_OVERRIDES = {
    "generations": "generations",
    "crossover_rate": "crossover_rate",
    "lasso_lambda": "lambda",
}

# hard wall-clock safety net (minutes) so a run always terminates even if the
# jar ignores the "generations" property above
MAX_MINUTES = 10


@dataclass
class MrgpConfig(ApproachConfig):
    name: str = "Mrgp"
    population_size: int = 500
    generations: int = 100
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    lasso_lambda: float = 0.01


class MrgpApproach(SR_approach):
    def __init__(self, population_size=500, generations=100, crossover_rate=0.7, mutation_rate=0.1,
                 lasso_lambda=0.01):
        self.cfg = MrgpConfig(population_size=population_size,
                               generations=generations,
                               crossover_rate=crossover_rate,
                               mutation_rate=mutation_rate,
                               lasso_lambda=lasso_lambda)
        super().__init__(self.cfg)
        self.sympy = []
        self.errors = []

        self.tourney_size = 10  # not part of the requested tuning grid; FlexGP's own example default
        self.function_set = "+ - * mydivide exp sin cos"

    def prepare(self):
        pass

    def _write_properties(self, path: str):
        lines = [
            "external_threads = 1",
            f"pop_size = {self.cfg.population_size}",
            "terminal_set = X1",
            f"function_set = {self.function_set}",
            f"tourney_size = {self.tourney_size}",
            f"mutation_rate = {self.cfg.mutation_rate}",
            f"{PROPERTY_NAME_OVERRIDES['generations']} = {self.cfg.generations}",
            f"{PROPERTY_NAME_OVERRIDES['crossover_rate']} = {self.cfg.crossover_rate}",
            f"{PROPERTY_NAME_OVERRIDES['lasso_lambda']} = {self.cfg.lasso_lambda}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _mrgp_eqn_to_sympy(self, eqn_str: str) -> sp.Expr:
        """Best-effort conversion of a raw mostAccurate.txt line into sympy.

        Assumes the documented function_set tokens (+ - * mydivide exp sin cos)
        appear as ordinary infix/function-call syntax, and that the single
        input variable is named X1 (1-indexed, per the terminal_set example
        in the FlexGP docs). This is unverified against real jar output.
        """
        s = eqn_str.strip()
        s = re.sub(r'\bX1\b', 'X_0', s)
        s = re.sub(r'\bx1\b', 'X_0', s)
        locals_map = {"mydivide": lambda a, b: a / (b + 1e-6)}
        return sp.sympify(s, locals=locals_map)

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        X = sr_evaluator.X
        y = sr_evaluator.y

        if not os.path.exists(MRGP_JAR_PATH):
            print(f"MRGP jar not found at {MRGP_JAR_PATH}; set MRGP_JAR_PATH. Iteration omitted")
            return

        run_dir = tempfile.mkdtemp(prefix="mrgp_run_")
        try:
            train_csv = os.path.join(run_dir, "train.csv")
            data = np.column_stack([X.reshape(-1, 1) if X.ndim == 1 else X, y])
            np.savetxt(train_csv, data, delimiter=",")

            props_path = os.path.join(run_dir, "mrgp.properties")
            self._write_properties(props_path)

            cmd = ["java", "-jar", os.path.abspath(MRGP_JAR_PATH),
                   "-train", train_csv, "-minutes", str(MAX_MINUTES), "-properties", props_path]
            subprocess.run(cmd, cwd=run_dir, capture_output=True, timeout=(MAX_MINUTES + 2) * 60)

            matches = glob.glob(os.path.join(run_dir, "**", "mostAccurate.txt"), recursive=True)
            if not matches:
                print("MRGP produced no mostAccurate.txt; iteration omitted")
                return
            with open(matches[0]) as f:
                raw_eqn = f.read().strip().splitlines()[-1]
        except Exception as e:
            print(f"MRGP run failed ({e}); iteration omitted")
            return
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        if sr_evaluator.should_stop:
            return

        try:
            expr = self._mrgp_eqn_to_sympy(raw_eqn)
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
            tokens, const_values = sympy_to_srtoolkit_tokens(sp.sympify('0'))
            sr_evaluator.models[tuple(tokens)] = ModelResult(expr=tokens, error=1000)
            self.errors.append(1000)
            self.sympy.append(sp.sympify('0'))

    @classmethod
    def from_config(self, config: dict) -> "MrgpApproach":
        return self(
            population_size=config.get("population_size", 500),
            generations=config.get("generations", 100),
            crossover_rate=config.get("crossover_rate", 0.7),
            mutation_rate=config.get("mutation_rate", 0.1),
            lasso_lambda=config.get("lasso_lambda", 0.01),
        )
