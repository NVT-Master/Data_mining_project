# src/evaluation/__init__.py
from .metrics import ModelMetrics
from .report import EvaluationReporter

# Backward-compatible alias
EvaluationReport = EvaluationReporter

__all__ = ['ModelMetrics', 'EvaluationReporter', 'EvaluationReport']
