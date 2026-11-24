"""Fake review detection: TF-IDF text features + writing-style features + logistic regression."""
from .model import Detector, train_model, load_or_train, DEFAULT_MODEL, DEFAULT_DATA

__all__ = ["Detector", "train_model", "load_or_train", "DEFAULT_MODEL", "DEFAULT_DATA"]
