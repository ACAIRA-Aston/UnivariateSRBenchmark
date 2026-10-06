from pathlib import Path
from typing import Dict, Tuple

# Directories and Paths
DATASET_FILES = {
    "Easy": Path("UnivariateSRBenchmark/Datasets/easy.parquet"),
    "Medium": Path("UnivariateSRBenchmark/Datasets/medium.parquet"),
    "Hard": Path("UnivariateSRBenchmark/Datasets/hard.parquet"),
}


# Default dataset folders where tuning best_run.json is stored
DEFAULT_TUNING_PROBLEMS = {
    "Easy": "Easy-9",
    "Medium": "Medium-0",
    "Hard": "Hard-9",
}

INTER_DATASET_FILES = {
    "Easy": Path("UnivariateSRBenchmark/Datasets/interpolation_easy.parquet"),
    "Medium": Path("UnivariateSRBenchmark/Datasets/interpolation_medium.parquet"),
    "Hard": Path("UnivariateSRBenchmark/Datasets/interpolation_hard.parquet"),
}

EXTRA_DATASET_FILES = {
    "Easy": Path("UnivariateSRBenchmark/Datasets/extrapolation_easy.parquet"),
    "Medium": Path("UnivariateSRBenchmark/Datasets/extrapolation_medium.parquet"),
    "Hard": Path("UnivariateSRBenchmark/Datasets/extrapolation_hard.parquet"),
}

# Registry for SR methods
APPROACH_REGISTRY: Dict[str, Tuple[str, str]] = {
    "Gplearn": ("UnivariateSRBenchmark.Methods.gplearn_approach", "GplearnApproach"),
    "Operon": ("UnivariateSRBenchmark.Methods.operon_approach", "OperonApproach"),
    "PySR": ("UnivariateSRBenchmark.Methods.pysr_approach", "PySRApproach"),
    "GPGOMEA": ("UnivariateSRBenchmark.Methods.gpgomea_approach", "GPGomeaApproach"),
    "ITEA": ("UnivariateSRBenchmark.Methods.itea_approach", "IteaApproach"),
    "DSR": ("UnivariateSRBenchmark.Methods.dsr_approach", "DSRApproach"),
    "UDSR": ("UnivariateSRBenchmark.Methods.udsr_approach", "UDSRApproach"),
    "E2E": ("UnivariateSRBenchmark.Methods.e2e_approach", "E2EApproach"),
    "TPSR": ("UnivariateSRBenchmark.Methods.tpsr_approach", "TPSRApproach"),
    "TaylorGP": ("UnivariateSRBenchmark.Methods.taylorgp_approach", "TaylorGPApproach"),
    "BSR": ("UnivariateSRBenchmark.Methods.bsr_approach", "BSRApproach"),
    "NeSymReS": ("UnivariateSRBenchmark.Methods.nesymres_approach", "NeSymReSApproach"),
    "AIFeynman": ("UnivariateSRBenchmark.Methods.aifeynman_approach", "AIFeynmanApproach"),
}