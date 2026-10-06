import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.UnivariateSRBenchmark.RunBench.run_benchmark import run_benchmark

if __name__ == '__main__':  #242E3

    run_benchmark(
        method="NeSymReS",
        tier="Easy",
        start_idx=0,
        end_idx=None,
    )
