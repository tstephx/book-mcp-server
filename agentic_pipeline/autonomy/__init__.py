"""Autonomy package for graduated trust management."""

from agentic_pipeline.autonomy.config import AutoApprovalDecision, AutonomyConfig
from agentic_pipeline.autonomy.metrics import MetricsCollector
from agentic_pipeline.autonomy.calibration import CalibrationEngine
from agentic_pipeline.autonomy.spot_check import SpotCheckManager

__all__ = [
    "AutoApprovalDecision",
    "AutonomyConfig",
    "MetricsCollector",
    "CalibrationEngine",
    "SpotCheckManager",
]
