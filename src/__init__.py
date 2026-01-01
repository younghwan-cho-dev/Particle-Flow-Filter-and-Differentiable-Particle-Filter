"""
Makes the directory importable.
Controls what gets exported when using from src import *
"""

from .prob_ssm import ProbSSM

__all__ = ['ProbSSM']