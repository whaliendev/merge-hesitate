import torch
import numpy as np
from tqdm import tqdm


def evaluate_model(
    model,
    dataloader,
    tokenizer,
    accelerator=None,
    max_length=512,
    num_beams=3,
    max_resolve_len=256,
):
    """
    Evaluate model on given dataloader and return metrics.

    Args:
        model: HesitateT5 model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        accelerator: Optional Accelerator for distributed evaluation
        max_length: Maximum sequence length for generation
        num_beams: Number of beams for beam search
        max_resolve_len: 解决方案的最大token数，生成时会使用2*max_resolve_len作为上限

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, coverage)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_confidences = []
    all_coverage = []

    # 只在主进程上显示进度条
    disable_progress_bar = (
        False if accelerator is None else not accelerator.is_main_process
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=disable_progress_bar):
            # Add generation parameters to batch
            batch["num_beams"] = num_beams
            batch["max_resolve_len"] = max_resolve_len

            # Generate solutions with confidence scores using unified batch interface
            generated, confidence = model.generate(**batch)

            # Decode generated text and targets
            for i in range(len(generated)):
                target_ids = batch["labels"][i]

                # Skip padding in target
                target_ids = target_ids[target_ids != tokenizer.pad_token_id]

                # Decode text
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

                # 将生成的文本和目标文本截断为max_resolve_len个词语
                gen_text_words = gen_text.split()[:max_resolve_len]
                target_text_words = target_text.split()[:max_resolve_len]
                gen_text = " ".join(gen_text_words)
                target_text = " ".join(target_text_words)

                # Check for empty generation (model declined to solve)
                is_empty = len(gen_text.strip()) == 0

                # Mark as covered if model provided a solution and confidence exceeds threshold
                is_covered = (
                    not is_empty and confidence[i] >= model.confidence_threshold
                )

                # 只有在置信度超过阈值时，才将预测视为模型给出的正式预测
                # 否则视为模型"拒绝回答"
                prediction = (
                    gen_text if confidence[i] >= model.confidence_threshold else ""
                )

                all_predictions.append(prediction)
                all_targets.append(target_text)
                all_confidences.append(confidence[i].item())
                all_coverage.append(is_covered)

    # 在分布式环境中收集所有进程的结果
    if accelerator is not None:
        all_predictions = accelerator.gather_for_metrics(all_predictions)
        all_targets = accelerator.gather_for_metrics(all_targets)
        all_confidences = accelerator.gather_for_metrics(all_confidences)
        all_coverage = accelerator.gather_for_metrics(all_coverage)

    # 只在主进程上计算指标
    if accelerator is None or accelerator.is_main_process:
        metrics = calculate_metrics(
            all_predictions, all_targets, all_confidences, all_coverage
        )

        # Log whether features were used
        if hasattr(model, "use_features"):
            metrics["use_features"] = model.use_features

    else:
        # 非主进程返回空字典，将在后面广播
        metrics = {}

    # 确保所有进程获得相同的指标结果
    if accelerator is not None:
        metrics = accelerator.gather(metrics)[0]  # 广播主进程的结果给所有进程

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
    # 检查输入是否为空
    if not predictions or not targets or len(predictions) != len(targets):
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
        }

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


def analyze_errors(
    predictions, targets, confidences, model_threshold=0.8, feature_values=None
):
    """
    Analyze error patterns to understand model weaknesses.

    Args:
        predictions: List of model predictions
        targets: List of target solutions
        confidences: List of confidence scores
        model_threshold: 模型的置信度阈值
        feature_values: Optional dict of feature values for each sample

    Returns:
        Error analysis summary
    """
    # 只考虑置信度大于阈值的预测
    error_indices = [
        i
        for i in range(len(predictions))
        if predictions[i] != targets[i] and confidences[i] >= model_threshold
    ]

    # High confidence errors (most concerning)
    high_conf_errors = [i for i in error_indices if confidences[i] >= 0.9]

    # Medium confidence errors
    med_conf_errors = [i for i in error_indices if 0.8 <= confidences[i] < 0.9]

    # Low confidence errors (near threshold)
    low_conf_errors = [
        i for i in error_indices if model_threshold <= confidences[i] < 0.8
    ]

    # Analyze feature patterns in errors if features provided
    feature_patterns = {}
    if feature_values:
        for feature_name, values in feature_values.items():
            # Compare feature distribution in errors vs overall
            error_values = [values[i] for i in error_indices]
            feature_patterns[feature_name] = {
                "mean_all": np.mean(values),
                "mean_errors": np.mean(error_values) if error_values else 0,
                "correlation_with_errors": (
                    np.corrcoef(
                        [1 if i in error_indices else 0 for i in range(len(values))],
                        values,
                    )[0, 1]
                    if len(set(values)) > 1
                    else 0
                ),
            }

    # Return error analysis
    return {
        "total_errors": len(error_indices),
        "high_conf_errors": len(high_conf_errors),
        "med_conf_errors": len(med_conf_errors),
        "low_conf_errors": len(low_conf_errors),
        "feature_patterns": feature_patterns,
    }
