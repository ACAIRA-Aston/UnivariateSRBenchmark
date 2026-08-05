import copy
import math
import numpy as np


def add(bound1, bound2=None):
    return [bound1[0] + bound2[0], bound1[1] + bound2[1]]


def sub(bound1, bound2=None):
    return [bound1[0] - bound2[1], bound1[1] - bound2[0]]


def mul(bound1, bound2=None):
    max_ = max(
        bound1[0] * bound2[0],
        bound1[0] * bound2[1],
        bound1[1] * bound2[0],
        bound1[1] * bound2[1],
    )
    min_ = min(
        bound1[0] * bound2[0],
        bound1[0] * bound2[1],
        bound1[1] * bound2[0],
        bound1[1] * bound2[1],
    )
    return [min_, max_]


def div(bound1, bound2=None):
    if bound1[0] > 0 and bound2[0] > 0:
        return [1e-10, 1e10]
    else:
        return [-1e10, 1e10]


def abs(bound1, bound2=None):
    low, high = bound1[0], bound1[1]
    if low <= 0 <= high:
        return [0.0, float(max(np.abs(low), np.abs(high)))]
    return [
        float(min(np.abs(low), np.abs(high))),
        float(max(np.abs(low), np.abs(high))),
    ]


def cosh(bound1, bound2=None):
    low = max(min(bound1[0], 100), -100)
    up = max(min(bound1[1], 100), -100)
    # Use np.abs() instead of built-in abs() to avoid calling the interval abs() function
    return [1.0, math.cosh(max(np.abs(low), np.abs(up)))]


def sinh(bound1, bound2=None):
    low = max(min(bound1[0], 100), -100)
    up = max(min(bound1[1], 100), -100)
    return [math.sinh(low), math.sinh(up)]


def tanh(bound1, bound2=None):
    low = max(min(bound1[0], 100), -100)
    up = max(min(bound1[1], 100), -100)
    return [math.tanh(low), math.tanh(up)]


def sin(bound1, bound2=None):
    return [-1, 1]


def cos(bound1, bound2=None):
    return [-1, 1]


def tan(bound1, bound2=None):
    return [-1e10, 1e10]


def asin(bound1, bound2=None):
    return [-np.pi / 2, np.pi / 2]


def acos(bound1, bound2=None):
    return [0, np.pi]


def atan(bound1, bound2=None):
    return [-np.pi / 2, np.pi / 2]


def asinh(bound1, bound2=None):
    return [-1e10, 1e10]


def acosh(bound1, bound2=None):
    return [0, 1e10]


def atanh(bound1, bound2=None):
    return [-1e10, 1e10]


def log(bound1, bound2=None):
    low_bound = max(1e-10, bound1[0])
    up_bound = max(1e-10, bound1[1])
    return [math.log(low_bound), math.log(up_bound)]


def exp(bound1, bound2=None):
    low = min(bound1[0], 100)
    up = min(bound1[1], 100)
    return [math.exp(low), math.exp(up)]


def sqrt(bound1, bound2=None):
    down = max(1e-10, bound1[0])
    up = max(1e-10, bound1[1])
    return [math.sqrt(down), math.sqrt(up)]


def cal_spacebound(function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt'], n_features_=1,
                   var_bound=[-5, 5], const_flag=True):
    # space = {}
    # log_bound, exp_bound, sin_bound, const_bound = [-1e+10, 1e+10], [1e-10, 1e+10], [-1, 1], [-1, 1]
    # _bound_map = {'add': log_bound,
    #               'sub': log_bound,
    #               'mul': log_bound,
    #               'div': log_bound,
    #               'sin': sin_bound,
    #               'cos': sin_bound,
    #               'log': log_bound,
    #               'exp': exp_bound,
    #               'sqrt': exp_bound,
    #               'tan': [-1e+10, 1e+10],
    #               'tanh': [-1e+10, 1e+10],
    #               'sinh': [-1e+10, 1e+10],
    #               'const': const_bound}
    space = {}

    unbounded = [-1e10, 1e10]
    positive_bound = [1e-10, 1e10]
    non_negative_bound = [0, 1e10]
    cosh_bound = [1, 1e10]
    unit_bound = [-1, 1]
    half_pi_bound = [-np.pi / 2, np.pi / 2]

    # Extract string names from gplearn custom objects or strings
    clean_function_set = [
        f.name if hasattr(f, 'name') else str(f) for f in function_set
    ]

    _bound_map = {
        'add': unbounded,
        'sub': unbounded,
        'mul': unbounded,
        'div': unbounded,
        'abs': non_negative_bound,
        'const': unit_bound,
        'log': unbounded,
        'exp': positive_bound,
        'sqrt': non_negative_bound,
        'sin': unit_bound,
        'cos': unit_bound,
        'tan': unbounded,
        'asin': half_pi_bound,
        'acos': [0, np.pi],
        'atan': half_pi_bound,
        'sinh': unbounded,
        'cosh': cosh_bound,
        'tanh': unit_bound,
        'asinh': unbounded,
        'acosh': non_negative_bound,
        'atanh': unbounded,
    }

    for i in range(len(var_bound) // 2):
        _bound_map.update({i: var_bound[2 * i: 2 * i + 2]})

    op_list = copy.deepcopy(clean_function_set)
    for i in range(n_features_):
        op_list.append(i)

    if const_flag:
        opNum = len(clean_function_set) + n_features_ + 1
        op_list.append('const')
    else:
        opNum = len(clean_function_set) + n_features_

    for op1 in range(opNum):
        for op2 in range(opNum):
            for op3 in range(opNum):
                if op1 < len(clean_function_set):
                    func_name = str(op_list[op1])
                    arg1 = op_list[op2]
                    arg2 = op_list[op3]

                    # Uniform 2-argument call across all functions
                    temp_bound = eval(
                        f'{func_name}(_bound_map[{repr(arg1)}], _bound_map[{repr(arg2)}])'
                    )
                    space[f'{func_name} {arg1} {arg2}'] = temp_bound

                elif op1 < len(clean_function_set) + n_features_:
                    space[f'{op_list[op1]} '] = _bound_map[op_list[op1]]

    return space


def select_space(space_bound, low_bound=0, high_bound=1, var_bound=[-1, 1]):
    selected_space = []
    for key, value in space_bound.items():  # key = space,value = bound
        if value[0] <= (low_bound + 0.1) and value[1] >= (high_bound - 0.1):
            selected_space.append(key)
    print(low_bound, high_bound, 'total selected space:', len(selected_space))
    return selected_space


if __name__ == '__main__':
    def _protected_exp(x):
        # Clip to avoid overflow/underflow floating point errors
        return np.exp(np.clip(x, -80, 80))


    def _protected_asin(x):
        # Domain: [-1, 1]
        return np.arcsin(np.clip(x, -1.0, 1.0))


    def _atan(x):
        # Domain: (-inf, inf)
        return np.arctan(x)


    def _protected_sinh(x):
        # Clip to avoid overflow in sinh
        return np.sinh(np.clip(x, -80, 80))


    def _protected_cosh(x):
        # Clip to avoid overflow in cosh
        return np.cosh(np.clip(x, -80, 80))


    def _tanh(x):
        # Domain: (-inf, inf), Output: [-1, 1]
        return np.tanh(x)


    def _asinh(x):
        # Domain: (-inf, inf)
        return np.arcsinh(x)


    def _protected_acosh(x):
        # Domain: [1, inf)
        return np.arccosh(np.maximum(1.0, x))


    def _protected_atanh(x):
        # Domain: (-1, 1)
        return np.arctanh(np.clip(x, -0.999, 0.999))


    from taylorGP.functions import make_function

    exp_func = make_function(function=_protected_exp, name='exp', arity=1)
    asin_func = make_function(function=_protected_asin, name='asin', arity=1)
    atan_func = make_function(function=_atan, name='atan', arity=1)
    sinh_func = make_function(function=_protected_sinh, name='sinh', arity=1)
    cosh_func = make_function(function=_protected_cosh, name='cosh', arity=1)
    tanh_func = make_function(function=_tanh, name='tanh', arity=1)
    asinh_func = make_function(function=_asinh, name='asinh', arity=1)
    acosh_func = make_function(function=_protected_acosh, name='acosh', arity=1)
    atanh_func = make_function(function=_protected_atanh, name='atanh', arity=1)

    function_set = [  # Built-in TaylorGP string operations:
        'add', 'sub', 'mul', 'div', 'sin', 'tan', 'log', 'sqrt', 'abs',

        # Custom wrapped objects:
        exp_func,
        asin_func,
        atan_func,
        sinh_func,
        cosh_func,
        tanh_func,
        asinh_func,
        acosh_func,
        atanh_func
    ]
    spacee = select_space(cal_spacebound(var_bound=[-5, 5], function_set=function_set))
