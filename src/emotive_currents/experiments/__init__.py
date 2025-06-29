"""
Emotion classification experiments.
"""

from .emotion_classifier import EmotionClassifierExperiment, run_experiment
from .emotion_labels import EMOTION_LABELS

__all__ = [
    "EmotionClassifierExperiment",
    "run_experiment", 
    "EMOTION_LABELS"
] 