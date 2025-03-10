import torch
import numpy as np
from tqdm import tqdm


def evaluate_model(model, dataloader, tokenizer, accelerator=None):
    """
    Evaluate model on given dataloader and return metrics.

    Args:
        model: HesitateT5 model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        accelerator: Optional Accelerator for distributed evaluation

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, coverage)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_confidences = []
    all_coverage = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Generate solutions with confidence scores
            generated, confidence = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4,
            )

            # Decode generated text and targets
            for i in range(len(generated)):
                target_ids = batch["labels"][i]

                # Skip padding in target
                target_ids = target_ids[target_ids != -100]

                # Decode text
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

                # Check for empty generation (model declined to solve)
                is_empty = len(gen_text.strip()) == 0

                # Mark as covered if model provided a solution
                is_covered = (
                    not is_empty and confidence[i] >= model.confidence_threshold
                )

                all_predictions.append(gen_text)
                all_targets.append(target_text)
                all_confidences.append(confidence[i].item())
                all_coverage.append(is_covered)

    # Gather results across all processes if using accelerator
    if accelerator is not None:
        all_predictions = accelerator.gather(all_predictions)
        all_targets = accelerator.gather(all_targets)
        all_confidences = accelerator.gather(all_confidences)
        all_coverage = accelerator.gather(all_coverage)

    # Calculate metrics
    metrics = calculate_metrics(
        all_predictions, all_targets, all_confidences, all_coverage
    )

    return metrics


def calculate_metrics(predictions, targets, confidences, coverage):
    """
    Calculate evaluation metrics.

    Args:
        predictions: List of model predictions
        targets: List of target solutions
        confidences: List of confidence scores
        coverage: List of boolean values indicating if model attempted solution

    Returns:
        Dictionary of metrics
    """
    # Calculate only for samples where model provided a solution
    covered_indices = [i for i, c in enumerate(coverage) if c]

    if not covered_indices:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
        }

    # Calculate accuracy of attempted solutions
    correct = [predictions[i] == targets[i] for i in covered_indices]

    # Calculate metrics
    accuracy = sum(correct) / len(correct) if correct else 0.0

    # Calculate precision, recall, F1
    # For our case, precision is accuracy on attempted solutions
    precision = accuracy

    # Recall considers all samples
    total_correct = sum([predictions[i] == targets[i] for i in range(len(predictions))])
    recall = total_correct / len(predictions) if predictions else 0.0

    # F1 score
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Coverage is percentage of samples where model attempted solution
    coverage_rate = len(covered_indices) / len(predictions) if predictions else 0.0

    # Return all metrics
    return {
        "accuracy": accuracy,  # Accuracy on attempted solutions
        "precision": precision,  # Same as accuracy in our case
        "recall": recall,  # Correct solutions / all samples
        "f1": f1,  # Harmonic mean of precision and recall
        "coverage": coverage_rate,  # Percentage of samples with solutions
    }


def analyze_errors(predictions, targets, confidences, feature_values=None):
    """
    Analyze error patterns to understand model weaknesses.

    Args:
        predictions: List of model predictions
        targets: List of target solutions
        confidences: List of confidence scores
        feature_values: Optional dict of feature values for each sample

    Returns:
        Error analysis summary
    """
    error_indices = [
        i
        for i in range(len(predictions))
        if predictions[i] != targets[i] and confidences[i] >= 0.5
    ]

    # High confidence errors (most concerning)
    high_conf_errors = [i for i in error_indices if confidences[i] >= 0.8]

    # Medium confidence errors
    med_conf_errors = [i for i in error_indices if 0.6 <= confidences[i] < 0.8]

    # Low confidence errors (least concerning since threshold would filter)
    low_conf_errors = [i for i in error_indices if 0.5 <= confidences[i] < 0.6]

    # Analyze feature patterns in errors if features provided
    feature_patterns = {}
    if feature_values:
        for feature_name, values in feature_values.items():
            # Compare feature distribution in errors vs overall
            error_values = [values[i] for i in error_indices]
            feature_patterns[feature_name] = {
                "mean_all": np.mean(values),
                "mean_errors": np.mean(error_values) if error_values else 0,
                "correlation_with_errors": np.corrcoef(
                    [1 if i in error_indices else 0 for i in range(len(values))], values
                )[0, 1]
                if len(set(values)) > 1
                else 0,
            }

    # Return error analysis
    return {
        "total_errors": len(error_indices),
        "high_conf_errors": len(high_conf_errors),
        "med_conf_errors": len(med_conf_errors),
        "low_conf_errors": len(low_conf_errors),
        "feature_patterns": feature_patterns,
    }
