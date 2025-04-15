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
        self.threshold = (
            initial_threshold  # Use initial_threshold to set the starting value
        )

        # For temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 2)

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

            # Use Adam optimizer instead of LBFGS
            optimizer = torch.optim.Adam([self.temperature], lr=0.01)
            num_steps = 100  # Number of optimization steps
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )

            # Standard optimization loop for Adam
            for _ in range(num_steps):
                optimizer.zero_grad()
                calibrated = self._temperature_scale(confidences)
                loss = nn.BCELoss()(calibrated, labels)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(
                        "Warning: NaN loss detected during temperature scaling. Stopping optimization."
                    )
                    # Reset temperature to default if optimization fails
                    self.temperature = nn.Parameter(torch.ones(1) * 2)
                    break

                loss.backward()
                optimizer.step()
                scheduler.step()
            else:  # Executed if loop completes without break
                final_loss = loss.item()
                print(
                    f"Temperature scaling finished. Final loss: {final_loss:.4f}, Final temp: {self.temperature.item():.4f}"
                )

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
        """Apply temperature scaling to probabilities by converting to logits first."""
        # Add a small epsilon for numerical stability when confidences are close to 0 or 1
        eps = 1e-6
        # Convert probabilities to logits
        logits = torch.logit(confidences, eps=eps)
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        # Convert scaled logits back to probabilities
        return torch.sigmoid(scaled_logits)

    def _optimize_threshold(self, confidences: List[float], correctness: List[bool]):
        """Find optimal confidence threshold to maximize F0.5 score, keeping previous threshold on failure."""
        # 处理极端情况: 如果没有数据或数据无用，则保留当前阈值并返回
        if not confidences or not correctness:
            return  # Keep previous threshold

        if all(correctness) or not any(correctness):
            return  # Keep previous threshold, data is not informative

        calibrated = self.calibrate(confidences)

        best_f1 = 0
        # Initialize best_threshold with the current threshold as fallback
        best_threshold = self.threshold

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

            # Use F0.5 score (beta=0.5) to prioritize precision
            beta = 0.5
            # Add epsilon for numerical stability in denominator
            denominator = (beta**2 * precision) + recall
            if denominator > 1e-9:  # Check if denominator is non-zero
                f_score = (1 + beta**2) * precision * recall / denominator
            else:
                f_score = 0.0  # Assign 0 if denominator is zero or too small

            # Check if this threshold gives a better F-score
            if f_score > best_f1:
                best_f1 = f_score
                best_threshold = t  # Update best_threshold only if improvement found

        # Update the threshold with the best one found (or the previous one if no improvement)
        self.threshold = best_threshold

    def get_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.threshold

    def set_threshold(self, threshold: float):
        """Manually set the confidence threshold."""
        self.threshold = threshold
