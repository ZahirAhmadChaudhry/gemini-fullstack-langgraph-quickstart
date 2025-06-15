"""
MiniLM + Linear-SVM classifier for Thematic Classification
--------------------------------------------------------
Light-weight model: uses the `all-MiniLM-L6-v2` multilingual sentence embeddings
(≈90 MB on disk) and a linear SVM on top.  Trains in <1 minute on CPU, no
fine-tuning required.
"""

from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)


class ThematicMiniLMSVMModel:
    """Sentence-Transformer embeddings + Linear-SVM."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', random_state: int = 42):
        # Use a lighter MiniLM variant and rely on the default HF cache layout
        # to avoid the incompatible directory structure returned when a custom
        # ``cache_folder`` argument is supplied (which broke older
        # ``sentence_transformers`` versions).
        self.model_name = model_name
        self.random_state = random_state

        # Initialize encoder with error handling for cache issues
        try:
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"Successfully initialized MiniLM-SVM with encoder {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            logger.info("Attempting to clear cache and retry...")
            try:
                # Try to clear cache and reinitialize
                import os
                import shutil
                from pathlib import Path

                # Get default cache directory
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
                if cache_dir.exists():
                    logger.info(f"Clearing cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)

                # Retry initialization
                self.encoder = SentenceTransformer(model_name)
                logger.info(f"Successfully initialized MiniLM-SVM after cache clear")
            except Exception as e2:
                logger.error(f"Failed to initialize even after cache clear: {e2}")
                raise RuntimeError(f"Cannot initialize SentenceTransformer: {e2}")

        self.clf: CalibratedClassifierCV | None = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)

    def train(self, X_train: List[str], y_train: List[int],
              X_val: List[str] | None = None, y_val: List[int] | None = None,
              C: float = 1.0) -> Dict[str, Any]:
        """Train SVM (with Platt calibration) on MiniLM embeddings."""
        logger.info("Encoding training texts with MiniLM …")
        X_emb = self._encode(X_train)

        base_svm = LinearSVC(C=C, class_weight='balanced', random_state=self.random_state)
        # Calibrate to get predict_proba
        self.clf = CalibratedClassifierCV(base_svm, cv=3)
        self.clf.fit(X_emb, y_train)
        self.is_trained = True

        train_pred = self.clf.predict(X_emb)
        train_acc = accuracy_score(y_train, train_pred)
        results = {'train_accuracy': train_acc}

        if X_val is not None and y_val is not None:
            val_acc = accuracy_score(y_val, self.clf.predict(self._encode(X_val)))
            results['val_accuracy'] = val_acc
            logger.info(f"MiniLM-SVM validation accuracy: {val_acc:.4f}")

        logger.info(f"MiniLM-SVM training completed – Train Acc: {train_acc:.4f}")
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.clf.predict(self._encode(texts))  # type: ignore[arg-type]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.clf.predict_proba(self._encode(texts))  # type: ignore[arg-type]

    def evaluate(self, X_test: List[str], y_test: List[int], class_names: List[str] | None = None):
        if not self.is_trained:
            raise ValueError("Model not trained")
        preds = self.predict(X_test)
        probas = self.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        conf = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        return {
            'accuracy': acc,
            'confusion_matrix': conf,
            'classification_report': report,
            'predictions': preds,
            'probabilities': probas,
        }

    def save_model(self, path: str):
        if not self.is_trained:
            raise ValueError("Train before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'svm': self.clf}, path)
        logger.info(f"MiniLM-SVM saved to {path}")

    def load_model(self, path: str):
        obj = joblib.load(path)
        self.clf = obj['svm']
        self.is_trained = True
        logger.info(f"MiniLM-SVM loaded from {path}") 