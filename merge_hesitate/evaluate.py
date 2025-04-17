import torch
import numpy as np
from tqdm import tqdm
from merge_hesitate.confidence import ConfidenceCalibrator # Make sure calibrator class is imported


def evaluate_model(
    model,
    dataloader,
    tokenizer,
    accelerator,
    stage="val", # Add stage parameter ('val' or 'test')
    calibrator: ConfidenceCalibrator | None = None, # Add optional calibrator argument
    max_length=512,
    num_beams=3,
    max_resolve_len=256,
):
    """
    Evaluate model on given dataloader and return metrics. Applies calibration if calibrator is provided (on main process).

    Args:
        model: HesitateT5 model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        accelerator: Accelerator for distributed evaluation
        stage: Evaluation stage ('val' or 'test')
        calibrator: Optional ConfidenceCalibrator object for calibrating scores before metric calculation (used on main process).
        max_length: Maximum sequence length for generation
        num_beams: Number of beams for beam search
        max_resolve_len: 解决方案的最大token数

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, coverage)
    """
    model.eval()
    all_raw_confidences = [] # Collect raw confidences
    all_exact_matches = [] # Stores bool: (exact generation match, ignoring confidence)
    total_loss = 0.0
    total_valid_tokens = 0

    pad_token_id = tokenizer.pad_token_id
    unwrapped_model = accelerator.unwrap_model(model) # Unwrap once for threshold access
    confidence_threshold = unwrapped_model.confidence_threshold

    # --- Debug ---
    printed_eval_samples = 0
    max_prints = 5
    # --- End Debug ---

    # Disable progress bar on non-main processes
    disable_progress_bar = not accelerator.is_main_process

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating ({stage})", disable=disable_progress_bar)):

            if stage == "val":
                # Validation uses model.forward
                batch["stage"] = "val"
                outputs = model(**batch)
                # batch_size, seq_len
                predictions = outputs["predictions"]
                # batch_size, seq_len
                raw_confidence = outputs["confidence"] # Get raw confidence
                # batch_size, 1
                labels = batch["labels"]
                # 1
                batch_loss = outputs["loss"]
                # 1
                batch_valid_tokens = outputs["valid_tokens"]

                # --- Debug Printing (uses raw confidence) ---
                if accelerator.is_main_process and printed_eval_samples < max_prints:
                    num_to_print_this_batch = min(max_prints - printed_eval_samples, predictions.size(0))
                    for i in range(num_to_print_this_batch):
                         accelerator.print("\n" + "-" * 20 + f" Eval Sample {printed_eval_samples} (Batch {batch_idx}, Idx {i}) " + "-" * 20)
                         # Convert to list for cleaner printing
                         pred_ids = predictions[i].tolist()
                         label_ids = labels[i].tolist()
                         # Optionally trim padding for readability
                         try:
                              pred_ids_trimmed = pred_ids[:pred_ids.index(pad_token_id)]
                         except ValueError:
                              pred_ids_trimmed = pred_ids
                         try:
                              label_ids_trimmed = label_ids[:label_ids.index(pad_token_id)]
                         except ValueError:
                              label_ids_trimmed = label_ids
                         
                         accelerator.print(f"Prediction IDs: {pred_ids_trimmed}")
                         accelerator.print(f"Target IDs    : {label_ids_trimmed}")
                         accelerator.print(f"Raw Confidence    : {raw_confidence[i].item():.4f}")
                         accelerator.print("-" * (42 + len(f" Eval Sample {printed_eval_samples} (Batch {batch_idx}, Idx {i}) ")) + "\n")
                         printed_eval_samples += 1
                # --- End Debug Printing ---

                # --- Vectorized Exact Match Calculation (Val) ---
                # Get predictions and original labels
                predictions = outputs["predictions"]
                original_labels = batch["labels"]

                # Ensure predictions and labels are of the same length
                batch_size, seq_len_pred = predictions.shape
                _, seq_len_label = original_labels.shape
                
                # Always adjust predictions to match label length
                if seq_len_pred < seq_len_label:
                    padding_size = seq_len_label - seq_len_pred
                    predictions = torch.cat([
                        predictions,
                        torch.full((batch_size, padding_size), pad_token_id, device=predictions.device, dtype=predictions.dtype)
                    ], dim=1)
                elif seq_len_pred > seq_len_label:
                    # Truncate predictions if longer than labels
                    predictions = predictions[:, :seq_len_label]
                
                # --- Calculate sequence-level exact match (与train.py保持一致) ---
                # 1. 内容匹配检查
                content_match = torch.all((predictions == original_labels) | (original_labels == pad_token_id), dim=1)

                # 2. 检查EOS位置的一致性
                eos_token_id = tokenizer.eos_token_id
                pred_has_eos = (predictions == eos_token_id).any(dim=1)
                label_has_eos = (original_labels == eos_token_id).any(dim=1)

                # 对于有EOS的序列，找EOS位置
                pred_eos_pos = torch.argmax((predictions == eos_token_id).float(), dim=1)
                label_eos_pos = torch.argmax((original_labels == eos_token_id).float(), dim=1)

                # 对于没有EOS的序列，找最后一个非pad token的位置
                pred_last_non_pad = torch.sum((predictions != pad_token_id).float(), dim=1) - 1
                label_last_non_pad = torch.sum((original_labels != pad_token_id).float(), dim=1) - 1
                # 确保不是负数 (如果全是pad)
                pred_last_non_pad = torch.maximum(pred_last_non_pad, torch.zeros_like(pred_last_non_pad))
                label_last_non_pad = torch.maximum(label_last_non_pad, torch.zeros_like(label_last_non_pad))

                # 根据是否有EOS选择使用哪个结束位置
                pred_end_pos = torch.where(pred_has_eos, pred_eos_pos, pred_last_non_pad)
                label_end_pos = torch.where(label_has_eos, label_eos_pos, label_last_non_pad)

                # 比较结束位置
                end_pos_match = pred_end_pos == label_end_pos

                # 检查EOS一致性(都有或都没有)
                eos_consistency = (pred_has_eos == label_has_eos)

                # 序列完全匹配需要三个条件: 内容匹配、边界匹配、EOS一致性
                exact_match = content_match & end_pos_match & eos_consistency
                # --- End Sequence-level Exact Match Calculation ---

                all_raw_confidences.append(raw_confidence)
                all_exact_matches.append(exact_match) # Stores bool (Exact Match Only)
                total_loss += batch_loss
                total_valid_tokens += batch_valid_tokens

            elif stage == "test":
                # Test uses model.generate
                gen_kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "features": batch.get("features"),
                    "num_beams": num_beams,
                    "max_resolve_len": max_resolve_len,
                }
                if not unwrapped_model.use_features:
                     gen_kwargs.pop("features")

                generated_ids, raw_confidence = unwrapped_model.generate(**gen_kwargs)
                
                original_labels_with_bos = batch["labels"]

                # Shift labels right for comparison with generated_ids
                labels_for_match = original_labels_with_bos.clone()
                pad_token_id = tokenizer.pad_token_id
                print("labels_for_match: \n", labels_for_match)

                # Need eos_token_id
                eos_token_id = tokenizer.eos_token_id

                # Pad shorter sequence to match the longer one for comparison
                batch_size, seq_len_pred = generated_ids.shape
                _, seq_len_label = labels_for_match.shape # Use shifted labels length
                seq_len = max(seq_len_pred, seq_len_label)

                # Always adjust predictions to match label length
                if seq_len_pred < seq_len_label:
                    padding_size = seq_len_label - seq_len_pred
                    generated_ids = torch.cat([
                        generated_ids,
                        torch.full((batch_size, padding_size), pad_token_id, device=generated_ids.device, dtype=generated_ids.dtype)
                    ], dim=1)
                elif seq_len_pred > seq_len_label:
                    # Truncate predictions if longer than labels
                    generated_ids = generated_ids[:, :seq_len_label]

                # --- Vectorized Exact Match Calculation (Test) ---
                # Ensure predictions and labels are of the same length
                batch_size, seq_len_pred = generated_ids.shape
                _, seq_len_label = original_labels_with_bos.shape
                
                # Always adjust predictions to match label length
                if seq_len_pred < seq_len_label:
                    padding_size = seq_len_label - seq_len_pred
                    generated_ids = torch.cat([
                        generated_ids,
                        torch.full((batch_size, padding_size), pad_token_id, device=generated_ids.device, dtype=generated_ids.dtype)
                    ], dim=1)
                elif seq_len_pred > seq_len_label:
                    # Truncate predictions if longer than labels
                    generated_ids = generated_ids[:, :seq_len_label]
                
                # --- Calculate sequence-level exact match (与train.py保持一致) ---
                # 1. 内容匹配检查
                content_match = torch.all((generated_ids == original_labels_with_bos) | (original_labels_with_bos == pad_token_id), dim=1)

                # 2. 检查EOS位置的一致性
                pred_has_eos = (generated_ids == eos_token_id).any(dim=1)
                label_has_eos = (original_labels_with_bos == eos_token_id).any(dim=1)

                # 对于有EOS的序列，找EOS位置
                pred_eos_pos = torch.argmax((generated_ids == eos_token_id).float(), dim=1)
                label_eos_pos = torch.argmax((original_labels_with_bos == eos_token_id).float(), dim=1)

                # 对于没有EOS的序列，找最后一个非pad token的位置
                pred_last_non_pad = torch.sum((generated_ids != pad_token_id).float(), dim=1) - 1
                label_last_non_pad = torch.sum((original_labels_with_bos != pad_token_id).float(), dim=1) - 1
                # 确保不是负数 (如果全是pad)
                pred_last_non_pad = torch.maximum(pred_last_non_pad, torch.zeros_like(pred_last_non_pad))
                label_last_non_pad = torch.maximum(label_last_non_pad, torch.zeros_like(label_last_non_pad))

                # 根据是否有EOS选择使用哪个结束位置
                pred_end_pos = torch.where(pred_has_eos, pred_eos_pos, pred_last_non_pad)
                label_end_pos = torch.where(label_has_eos, label_eos_pos, label_last_non_pad)

                # 比较结束位置
                end_pos_match = pred_end_pos == label_end_pos

                # 检查EOS一致性(都有或都没有)
                eos_consistency = (pred_has_eos == label_has_eos)

                # 序列完全匹配需要三个条件: 内容匹配、边界匹配、EOS一致性
                exact_match = content_match & end_pos_match & eos_consistency
                # --- End Sequence-level Exact Match Calculation ---

                all_raw_confidences.append(raw_confidence)
                all_exact_matches.append(exact_match) # Stores bool (Exact Match Only)

                # --- Debug Printing (Test Stage) ---
                if accelerator.is_main_process and printed_eval_samples < max_prints:
                    num_to_print_this_batch = min(max_prints - printed_eval_samples, generated_ids.size(0))
                    for i in range(num_to_print_this_batch):
                         accelerator.print("\n" + "-" * 20 + f" Test Sample {printed_eval_samples} (Batch {batch_idx}, Idx {i}) " + "-" * 20)
                         # Convert to list for cleaner printing
                         gen_ids = generated_ids[i].tolist()
                         # Use SHIFTED labels for debug comparison to align with what was checked
                         label_ids_test = labels_for_match[i].tolist()
                         # Optionally trim padding for readability
                         try:
                              gen_ids_trimmed = gen_ids[:gen_ids.index(pad_token_id)]
                         except ValueError:
                              gen_ids_trimmed = gen_ids # No pad found
                         try:
                              eos_index = gen_ids_trimmed.index(tokenizer.eos_token_id)
                              gen_ids_trimmed = gen_ids_trimmed[:eos_index]
                         except ValueError:
                             pass # No EOS found

                         try:
                              label_ids_trimmed_test = label_ids_test[:label_ids_test.index(pad_token_id)]
                         except ValueError:
                              label_ids_trimmed_test = label_ids_test

                         accelerator.print(f"Generated IDs : {gen_ids_trimmed}")
                         accelerator.print(f"Target IDs (Shifted) : {label_ids_trimmed_test}") # Indicate labels are shifted
                         # Decode for readability
                         accelerator.print(f"Generated Text: {tokenizer.decode(gen_ids_trimmed, skip_special_tokens=False)}") # Keep special tokens for debugging?
                         accelerator.print(f"Target Text (Shifted): {tokenizer.decode(label_ids_trimmed_test, skip_special_tokens=False)}")
                         accelerator.print(f"Raw Confidence: {raw_confidence[i].item():.4f}") # Show raw confidence for this generated sample
                         accelerator.print("-" * (42 + len(f" Test Sample {printed_eval_samples} (Batch {batch_idx}, Idx {i}) ")) + "\n")
                         printed_eval_samples += 1
                # --- End Debug Printing (Test Stage) ---

            else:
                raise ValueError(f"Invalid stage for evaluation: {stage}. Must be 'val' or 'test'.")

    # Concatenate results from all batches
    all_raw_confidences = torch.cat(all_raw_confidences)
    all_exact_matches = torch.cat(all_exact_matches)

    # Gather results from all processes
    gathered_confidences = accelerator.gather_for_metrics(all_raw_confidences)
    gathered_exact_matches = accelerator.gather_for_metrics(all_exact_matches)

    if stage == "val":
        # Gather loss and valid tokens only for validation stage
        total_loss_tensor = torch.tensor(float(total_loss), dtype=torch.float, device=accelerator.device) # Cast to float
        total_valid_tokens_tensor = torch.tensor(float(total_valid_tokens), dtype=torch.float, device=accelerator.device) # Cast to float
        gathered_losses = accelerator.gather(total_loss_tensor)
        gathered_valid_tokens = accelerator.gather(total_valid_tokens_tensor)
        # Sum on main process
        if accelerator.is_main_process:
            total_loss = torch.sum(gathered_losses).item()
            total_valid_tokens = torch.sum(gathered_valid_tokens).item()
        else:
             total_loss = 0
             total_valid_tokens = 1 # Avoid division by zero

    # Calculate final metrics on the main process
    metrics = {}
    if accelerator.is_main_process:
        num_samples = len(gathered_exact_matches)
        if num_samples == 0:
             # Handle empty evaluation case
             precision = 0.0
             recall = 0.0
             f1 = 0.0
             coverage = 0.0
             avg_calibrated_confidence = 0.0
             if stage == "val":
                 avg_loss = 0.0
        else:
            # --- Apply Calibration on Main Process ---
            final_confidences_to_use = gathered_confidences # Default to raw
            if calibrator is not None:
                try:
                    # Calibrate requires list of floats
                    calibrated_list = calibrator.calibrate(gathered_confidences.cpu().tolist())
                    final_confidences_to_use = torch.tensor(calibrated_list, dtype=torch.float, device=gathered_confidences.device)
                    print(f"Applied calibration using {calibrator.strategy} strategy for {stage} evaluation.")
                except Exception as e:
                    print(f"Warning: Calibration failed during {stage} evaluation: {e}. Using raw scores.")
                    # Fallback to raw scores if calibration fails
                    final_confidences_to_use = gathered_confidences
            else:
                print(f"No calibrator provided for {stage} evaluation. Using raw scores.")
            # --- End Calibration ---

            # --- Calculate Metrics using potentially calibrated confidences ---
            confident_mask = final_confidences_to_use >= confidence_threshold
            num_confident = confident_mask.sum().item()

            # Correct predictions = Exact Match AND Confident (using calibrated confidence)
            correct_and_confident = gathered_exact_matches & confident_mask
            num_correct_and_confident = correct_and_confident.sum().item()

            # Precision: (Correct & Confident) / Confident
            precision = num_correct_and_confident / num_confident if num_confident > 0 else 0.0

            # Recall: (Correct & Confident) / Total Samples
            recall = num_correct_and_confident / num_samples

            # F1 Score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Coverage: Confident / Total Samples
            coverage = num_confident / num_samples

            # Average CALIBRATED confidence over all samples (if calibrated)
            avg_calibrated_confidence = final_confidences_to_use.mean().item()

            # Average loss for validation stage
            if stage == "val":
                avg_loss = total_loss / (total_valid_tokens + 1e-9)
            # --- End Metric Calculation ---

        # Populate metrics dictionary
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["coverage"] = coverage
        metrics["avg_confidence"] = avg_calibrated_confidence # Report avg calibrated confidence
        metrics["num_samples"] = num_samples
        if stage == "val":
            metrics["avg_loss"] = avg_loss

    # Broadcast metrics from main process to others if needed
    # metrics = accelerator.broadcast(metrics, from_process=0) # Usually not needed unless other processes act on metrics

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
