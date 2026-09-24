"""
General evaluation utilities for generative models trained for
particle-shower generation.

Author:
    Diptarko Choudhury

Created:
    17 September 2026

Contact:
    cdiptarko@gmail.com
"""

from LazyEval.core.evaluator import Dataset, EvaluationModel, Infer
from LazyEval.core.util import process_in_batch_and_return as ProcessInBatch
from LazyEval.core.loader import LazyShowerDataLoader as LazyShowerData

__all__ = [
    "Dataset",
    "EvaluationModel",
    "Infer",
    "ProcessInBatch",
    "LazyShowerData"
]