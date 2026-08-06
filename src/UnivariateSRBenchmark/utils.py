import sympy as sp
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def sympy_to_srtoolkit_tokens(expr, constants=None):
    if constants is None:
        constants = []

    def rec(e):
        # if e is sp.I:
        #     return ["i"]
        if e.is_Symbol:
            return [str(e)]
        if e is sp.pi:
            return ["pi"]
        if e is sp.E:
            return ["e"]
        if e.is_Number:
            constants.append(float(e))
            return [str(e)]
        if e.is_Add:
            args = e.as_ordered_terms()
            toks = ["("]
            for i, a in enumerate(args):
                if i > 0:
                    if a.could_extract_minus_sign():
                        toks += ["-"]; a = -a
                    else:
                        toks += ["+"]
                toks += rec(a)
            toks.append(")")
            return toks
        if e.is_Mul:
            # detect unary minus on a whole product, e.g. neg(...) surviving as -1*(...)
            coeff, rest = e.as_coeff_Mul()
            if coeff == -1 and rest != 1:
                return ["(", "-"] + rec(rest) + [")"]
            args = e.as_ordered_factors()
            toks = ["("]
            for i, a in enumerate(args):
                if i > 0:
                    toks.append("*")
                toks += rec(a)
            toks.append(")")
            return toks
        if e.is_Pow:
            base, exp_ = e.args
            if exp_ == -1:
                return rec(base) + ["^-1"]
            if exp_ == 2:
                return rec(base) + ["^2"]
            if exp_ == 3:
                return rec(base) + ["^3"]
            return ["("] + rec(base) + ["^"] + rec(exp_) + [")"]
        if isinstance(e, sp.sin):
            return ["sin", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.cos):
            return ["cos", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.tan):
            return ["tan", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.exp):
            return ["exp", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.log):
            return ["ln", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.Abs):
            return ["abs", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.sinh):
            return ["sinh", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.cosh):
            return ["cosh", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.tanh):
            return ["tanh", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.asin):
            return ["asin", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.acos):
            return ["acos", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.atan):
            return ["atan", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.asinh):
            return ["asinh", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.acosh):
            return ["acosh", "("] + rec(e.args[0]) + [")"]
        if isinstance(e, sp.atanh):
            return ["atanh", "("] + rec(e.args[0]) + [")"]
        raise ValueError(f"Unsupported node: {e} ({type(e)})")

    return rec(expr), constants
