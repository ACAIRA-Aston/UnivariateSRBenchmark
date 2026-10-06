import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.UnivariateSRBenchmark.Tuning.grid_search import run_grid_search

if __name__ == '__main__':

    # For the Easy tier, use 'Easy-9'
    # For the Medium tier, use 'Medium-0'
    # For the Hard tier, use 'Hard-9'
    dtaset = 'Hard-9'
    print(dtaset)
    results = run_grid_search("UDSR", dtaset)

