import argparse
import os
import sys

from src.taylorGP.genetic import SymbolicRegressor

from sklearn.metrics import mean_squared_error  # 均方误差
import pandas as pd
from time import  time
import random
from sklearn.model_selection import train_test_split


est = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                        generations=500, stopping_criteria=1e-10,
                        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'tanh', 'log', 'exp', 'sqrt'],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.3,
                        max_samples=1.0, verbose=0,
                        parsimony_coefficient=0.1,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), low_memory=True,
                        max_time = 3600-10)

import numpy as np
n = 10000
np.random.seed(4)
x1 = np.random.uniform(0, 20, size=n)
x2 = np.random.uniform(-5, 5, size=n)
X = np.array([x1, x2]).T
# Calculate output
Y = np.sqrt(x1) * np.log(x2 ** 2)
est.fit(X[:10000], Y[:10000])
print(est.sympy_global_best)


est = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                        generations=500, stopping_criteria=1e-10,
                        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'tanh', 'log', 'exp', 'sqrt'],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.3,
                        max_samples=1.0, verbose=0,
                        parsimony_coefficient=0.1,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), low_memory=True,
                        max_time = 3600-10)

np.random.seed(5)
x1 = np.random.uniform(0, 20, size=n)
x2 = np.random.uniform(-5, 5, size=n)
X = np.array([x1, x2]).T
# Calculate output
Y = np.sqrt(x1) * np.log(x2 ** 2)
est.fit(X[:10000], Y[:10000])
print(est.sympy_global_best)

est = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                        generations=500, stopping_criteria=1e-10,
                        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'tanh', 'log', 'exp', 'sqrt'],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.3,
                        max_samples=1.0, verbose=0,
                        parsimony_coefficient=0.1,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), low_memory=True,
                        max_time = 3600-10)

np.random.seed(6)
x1 = np.random.uniform(0, 20, size=n)
x2 = np.random.uniform(-5, 5, size=n)
X = np.array([x1, x2]).T
# Calculate output
Y = np.sqrt(x1) * np.log(x2 ** 2)
est.fit(X[:10000], Y[:10000])
print(est.sympy_global_best)
