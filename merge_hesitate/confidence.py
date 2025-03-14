import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import List


class ConfidenceCalibrator:
    """
    Calibrate model confidence scores to better reflect true correctness probabilities.
    """

    def __init__(self, strategy="temperature", initial_threshold=0.8):
        self.strategy = strategy
        self.calibration_model = None
        self.threshold = initial_threshold

        # For temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, raw_confidences: List[float], correctness: List[bool]):
        """
        Fit calibration model based on raw confidence scores and actual correctness.

        Args:
            raw_confidences: Uncalibrated confidence scores from the model
            correctness: Whether the model prediction was correct
        """
        if len(raw_confidences) != len(correctness):
            raise ValueError(
                "Confidence scores and correctness lists must have the same length"
            )

        if self.strategy == "isotonic":
            self.calibration_model = IsotonicRegression(out_of_bounds="clip")
            self.calibration_model.fit(
                raw_confidences, [1.0 if c else 0.0 for c in correctness]
            )

        elif self.strategy == "temperature":
            # Convert to PyTorch tensors
            confidences = torch.tensor(raw_confidences, dtype=torch.float)
            labels = torch.tensor(
                [1.0 if c else 0.0 for c in correctness], dtype=torch.float
            )

            # Optimize temperature parameter
            optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

            def closure():
                optimizer.zero_grad()
                calibrated = self._temperature_scale(confidences)
                loss = nn.BCELoss()(calibrated, labels)
                loss.backward()
                return loss

            optimizer.step(closure)

        # Find optimal threshold
        self._optimize_threshold(raw_confidences, correctness)

    def calibrate(self, raw_confidences: List[float]) -> List[float]:
        """Apply calibration to raw confidence scores."""
        if self.strategy == "isotonic" and self.calibration_model is not None:
            return self.calibration_model.predict(raw_confidences).tolist()

        elif self.strategy == "temperature":
            confidences = torch.tensor(raw_confidences, dtype=torch.float)
            return self._temperature_scale(confidences).tolist()

        return raw_confidences  # Return uncalibrated if no model is fit

    def _temperature_scale(self, confidences: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to raw confidence scores."""
        return torch.sigmoid(confidences / self.temperature)

    def _optimize_threshold(self, confidences: List[float], correctness: List[bool]):
        """Find optimal confidence threshold to maximize accuracy."""
        calibrated = self.calibrate(confidences)

        best_f1 = 0
        best_threshold = 0.5

        # Try different thresholds
        for t in np.linspace(0.5, 0.99, 50):
            # Calculate weighted F1 score (more weight on precision)
            predictions = [c >= t for c in calibrated]

            # Only consider cases where we would make a prediction
            predicted_indices = [i for i, p in enumerate(predictions) if p]

            if not predicted_indices:
                continue

            correct_predictions = sum(correctness[i] for i in predicted_indices)
            precision = (
                correct_predictions / len(predicted_indices) if predicted_indices else 0
            )
            recall = (
                correct_predictions / sum(correctness) if sum(correctness) > 0 else 0
            )

            # Use F2 score to prioritize precision over recall (beta=0.5)
            beta = 0.5
            f_score = (
                (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
                if (precision + recall) > 0
                else 0
            )

            if f_score > best_f1:
                best_f1 = f_score
                best_threshold = t

        self.threshold = best_threshold

    def get_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.threshold

    def set_threshold(self, threshold: float):
        """Manually set the confidence threshold."""
        self.threshold = threshold
