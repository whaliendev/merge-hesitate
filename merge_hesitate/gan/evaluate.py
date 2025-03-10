import torch
import numpy as np
from tqdm import tqdm

def evaluate_gan_model(gan_model, dataloader, tokenizer, accelerator=None):
    """
    Evaluate GAN model on given dataloader and return metrics.
    
    Args:
        gan_model: MergeHesitateGAN model
        dataloader: Evaluation dataloader
        tokenizer: Tokenizer for decoding
        accelerator: Optional Accelerator for distributed evaluation
    
    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1, coverage)
    """
    gan_model.eval()
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_refined_confidences = []
    all_coverage = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating GAN"):
            # Generate solutions with refined confidence scores
            generated, refined_confidence = gan_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4
            )
            
            # Get raw confidence (from generator only)
            _, raw_confidence = gan_model.generator.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4
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
                is_covered = not is_empty and refined_confidence[i] >= gan_model.confidence_threshold
                
                all_predictions.append(gen_text)
                all_targets.append(target_text)
                all_confidences.append(raw_confidence[i].item())
                all_refined_confidences.append(refined_confidence[i].item())
                all_coverage.append(is_covered)
    
    # Gather results across all processes if using accelerator
    if accelerator is not None:
        all_predictions = accelerator.gather(all_predictions)
        all_targets = accelerator.gather(all_targets)
        all_confidences = accelerator.gather(all_confidences)
        all_refined_confidences = accelerator.gather(all_refined_confidences)
        all_coverage = accelerator.gather(all_coverage)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_predictions, 
        all_targets, 
        all_refined_confidences, 
        all_coverage
    )
    
    # Add metrics comparing raw and refined confidence
    metrics.update(compare_confidence_metrics(
        all_predictions, 
        all_targets, 
        all_confidences, 
        all_refined_confidences
    ))
    
    return metrics

def calculate_metrics(predictions, targets, confidences, coverage):
    """Calculate evaluation metrics."""
    # Only consider covered examples for accuracy
    covered_indices = [i for i, covered in enumerate(coverage) if covered]
    
    # Covered predictions and targets
    covered_preds = [predictions[i] for i in covered_indices]
    covered_targets = [targets[i] for i in covered_indices]
    
    # Calculate accuracy on covered examples
    correct = [p == t for p, t in zip(covered_preds, covered_targets)]
    accuracy = sum(correct) / len(correct) if covered_preds else 0.0
    
    # Calculate precision (same as accuracy in our case)
    precision = accuracy
    
    # Calculate recall (correct solutions / all samples)
    all_correct = [predictions[i] == targets[i] and coverage[i] for i in range(len(predictions))]
    recall = sum(all_correct) / len(targets) if targets else 0.0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate coverage rate
    coverage_rate = len(covered_indices) / len(predictions) if predictions else 0.0
    
    # Return all metrics
    return {
        "accuracy": accuracy,  # Accuracy on attempted solutions
        "precision": precision,  # Same as accuracy in our case
        "recall": recall,  # Correct solutions / all samples
        "f1": f1,  # Harmonic mean of precision and recall
        "coverage": coverage_rate,  # Percentage of samples with solutions
    }

def compare_confidence_metrics(predictions, targets, raw_confidences, refined_confidences):
    """Compare metrics related to raw vs refined confidence."""
    # Sort by confidence
    raw_sorted = sorted(
        list(zip(predictions, targets, raw_confidences)), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    refined_sorted = sorted(
        list(zip(predictions, targets, refined_confidences)), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    # Calculate accuracy for top N% of confident predictions
    percentiles = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    raw_accuracies = {}
    refined_accuracies = {}
    
    for p in percentiles:
        k = int(len(predictions) * p)
        if k == 0:
            continue
            
        # Top k by raw confidence
        raw_top_k = raw_sorted[:k]
        raw_correct = sum(pred == tgt for pred, tgt, _ in raw_top_k)
        raw_accuracies[f"raw_top_{int(p*100)}%"] = raw_correct / k
        
        # Top k by refined confidence
        refined_top_k = refined_sorted[:k]
        refined_correct = sum(pred == tgt for pred, tgt, _ in refined_top_k)
        refined_accuracies[f"refined_top_{int(p*100)}%"] = refined_correct / k
    
    # Combine metrics
    comparative = {}
    comparative.update(raw_accuracies)
    comparative.update(refined_accuracies)
    
    # Calculate confidence calibration error
    raw_calibration_error = calculate_calibration_error(predictions, targets, raw_confidences)
    refined_calibration_error = calculate_calibration_error(predictions, targets, refined_confidences)
    
    comparative["raw_calibration_error"] = raw_calibration_error
    comparative["refined_calibration_error"] = refined_calibration_error
    
    return comparative

def calculate_calibration_error(predictions, targets, confidences, num_bins=10):
    """Calculate expected calibration error."""
    bin_size = 1.0 / num_bins
    bins = {}
    
    # Assign predictions to confidence bins
    for pred, tgt, conf in zip(predictions, targets, confidences):
        bin_idx = min(int(conf / bin_size), num_bins - 1)
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append((pred, tgt, conf))
    
    # Calculate calibration error
    ece = 0.0
    total_samples = len(predictions)
    
    for bin_idx in range(num_bins):
        if bin_idx not in bins:
            continue
            
        bin_samples = bins[bin_idx]
        bin_size = len(bin_samples)
        bin_weight = bin_size / total_samples
        
        # Calculate accuracy and average confidence in this bin
        bin_correct = sum(pred == tgt for pred, tgt, _ in bin_samples)
        bin_accuracy = bin_correct / bin_size if bin_size > 0 else 0
        bin_confidence = sum(conf for _, _, conf in bin_samples) / bin_size
        
        # Add weighted absolute difference to ECE
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece

def analyze_gan_errors(gan_model, dataloader, tokenizer, accelerator=None):
    """
    Analyze error patterns to understand GAN model weaknesses.
    """
    gan_model.eval()
    
    errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing Errors"):
            # Generate solutions
            generated, refined_confidence = gan_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4
            )
            
            # Get raw confidence scores from generator
            _, raw_confidence = gan_model.generator.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4
            )
            
            # Get discriminator scores
            disc_scores = gan_model.discriminator(
                conflict_ids=batch["input_ids"],
                solution_ids=generated,
                attention_mask=batch["attention_mask"]
            )
            
            # Get detailed quality scores
            quality_scores = gan_model.discriminator.compute_detailed_scores(
                conflict_ids=batch["input_ids"],
                solution_ids=generated,
                attention_mask=batch["attention_mask"]
            )
            
            # Decode texts
            for i in range(len(generated)):
                target_ids = batch["labels"][i]
                target_ids = target_ids[target_ids != -100]
                
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                
                is_covered = refined_confidence[i] >= gan_model.confidence_threshold
                is_correct = gen_text == target_text
                
                # If covered but incorrect, this is an error
                if is_covered and not is_correct:
                    conflict_text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                    
                    errors.append({
                        "conflict": conflict_text,
                        "prediction": gen_text,
                        "target": target_text,
                        "raw_confidence": raw_confidence[i].item(),
                        "refined_confidence": refined_confidence[i].item(),
                        "discriminator_score": disc_scores[i].item(),
                        "quality_scores": quality_scores[i].tolist(),
                    })
    
    # Gather errors across processes
    if accelerator is not None:
        all_errors = accelerator.gather(errors)
    else:
        all_errors = errors
    
    # Analyze patterns in errors
    error_analysis = {
        "total_errors": len(all_errors),
        "high_conf_errors": sum(1 for e in all_errors if e["refined_confidence"] >= 0.9),
        "disc_vs_gen_disagreement": sum(1 for e in all_errors 
                                      if abs(e["raw_confidence"] - e["discriminator_score"]) >= 0.3),
        "avg_quality_scores": {
            "overall": sum(e["quality_scores"][0] for e in all_errors) / len(all_errors) if all_errors else 0,
            "syntactic": sum(e["quality_scores"][1] for e in all_errors) / len(all_errors) if all_errors else 0,
            "semantic": sum(e["quality_scores"][2] for e in all_errors) / len(all_errors) if all_errors else 0,
            "completeness": sum(e["quality_scores"][3] for e in all_errors) / len(all_errors) if all_errors else 0,
        }
    }
    
    return error_analysis, all_errors 