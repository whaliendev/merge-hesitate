import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import time
from torch.utils.data import DataLoader
from merge_hesitate.model import HesitateT5
from merge_hesitate.data import create_dataloaders
from merge_hesitate.confidence import ConfidenceCalibrator
from merge_hesitate.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def train(args):
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        project_dir=args.output_dir # Specify project dir for logs/checkpoints
    )

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Ensure special tokens are added to tokenizer if they aren't already
    special_tokens = ["<lbra>", "<mbra>", "<rbra>"]

    # Add special tokens if they don't exist
    successed_num = tokenizer.add_tokens(special_tokens)
    assert successed_num == len(special_tokens), "Failed to add special tokens"

    # Log feature usage setting
    if args.use_features:
        accelerator.print(
            f"Feature extraction ENABLED with {args.feature_size} features"
        )
    else:
        accelerator.print("Feature extraction DISABLED")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_resolve_length=args.max_resolve_len,
        feature_size=args.feature_size,
        use_features=args.use_features,
        accelerator=accelerator,
    )

    # Initialize model
    model = HesitateT5(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        confidence_threshold=args.initial_threshold,
        feature_size=args.feature_size,
        use_features=args.use_features,
    )

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Calculate total training steps
    total_steps = (
        len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Prepare for distributed training
    # Note: Don't prepare test_loader here, prepare it before final evaluation
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Define loss functions
    conf_criterion = nn.BCELoss()

    # Training loop
    accelerator.print("***** Starting training *****")
    global_step = 0
    best_f1 = 0.0
    best_precision = 0.0
    last_calibrator = None # Store the most recent calibrator (only on main process)

    # Define path for the best model state
    best_model_state_path = os.path.join(args.output_dir, "best_model_state")
    best_calibrator_path = os.path.join(best_model_state_path, "calibrator.pt") # Define path

    # Log file paths (only main process needs them for writing)
    if accelerator.is_main_process:
        train_log_path = os.path.join(args.output_dir, "train_metrics.log")
        test_log_path = os.path.join(args.output_dir, "test_metrics.log")
        time_log_path = os.path.join(args.output_dir, "time_log.txt")
        # Clear existing log files or create headers if needed
        with open(train_log_path, "w") as f:
            # Write header for train metrics
            f.write("Epoch\tTrain Loss\tVal Metrics\tConfidence Threshold\tCalibrated\n")
        with open(test_log_path, "w") as f:
            # Write header for test metrics
            f.write("Test Metrics\n")
        with open(time_log_path, "w") as f:
            # Write header for timing info
            f.write("Timing Information\n")

    # Record start time
    start_time = time.time()

    # tqdm progress bar setup
    disable_progress_bar = not accelerator.is_main_process

    ever_calibrated = False
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", disable=disable_progress_bar
        )

        for step, batch in enumerate(progress_bar):
            # Add stage to batch
            batch["stage"] = "train"

            # Forward pass with unified batch interface
            # The forward pass now expects labels and returns loss, confidence, logits, valid_tokens
            outputs = model(**batch)

            # Get loss and other metrics
            # gen_loss is already the SUMMED NLL loss for the batch from model.forward
            # We normalize by the number of valid (non-padded) tokens
            gen_loss = outputs["loss"] / (outputs["valid_tokens"] + 1e-9) # Avoid division by zero
            # size: [batch_size]
            confidence = outputs["confidence"]
            # Logits size: [batch_size, seq_len, vocab_size]
            logits = outputs["logits"]

            # --- Corrected Vectorized Correctness Calculation ---
            with torch.no_grad():
                # Get predictions from logits [batch_size, seq_len, vocab_size] -> [batch_size, seq_len]
                # predictions[b, t] is prediction for original label token t+1
                predictions = torch.argmax(logits, dim=-1)

                # Need pad_token_id and eos_token_id
                pad_token_id = tokenizer.pad_token_id
                eos_token_id = tokenizer.eos_token_id

                target_labels = batch["labels"]

                # Pad predictions to match label length
                batch_size, seq_len_pred = predictions.shape
                _, seq_len_labels = target_labels.shape
                
                # Always pad predictions to match label length
                if seq_len_pred < seq_len_labels:
                    padding_size = seq_len_labels - seq_len_pred
                    predictions = torch.cat([
                        predictions,
                        torch.full((batch_size, padding_size), pad_token_id, device=predictions.device, dtype=predictions.dtype)
                    ], dim=1)
                elif seq_len_pred > seq_len_labels:
                    # Truncate predictions if longer than labels
                    predictions = predictions[:, :seq_len_labels]

                # Now predictions and target_labels have shape [batch_size, seq_len]
                # and target_labels[b, t] is the original label token t+1

                # 1. Create prediction mask (True up to and including first EOS)
                pred_indices = torch.arange(seq_len_labels, device=predictions.device).unsqueeze(0).expand(batch_size, -1)
                eos_mask_pred = (predictions == eos_token_id)
                first_eos_idx_pred = torch.where(eos_mask_pred.any(dim=1), eos_mask_pred.float().argmax(dim=1), seq_len_labels)
                pred_mask = pred_indices <= first_eos_idx_pred.unsqueeze(1)

                # 2. Create label mask (True up to PAD) - Use the SHIFTED target_labels
                label_mask = (target_labels != pad_token_id)

                # 3. Combine masks
                combined_mask = pred_mask & label_mask

                # 4. Calculate correct tokens using the combined mask and SHIFTED target_labels
                correct_tokens = (predictions == target_labels) & combined_mask

                # 5. Calculate num_correct_tokens and num_valid_tokens based on combined_mask
                num_correct_tokens = correct_tokens.sum(dim=1)
                num_valid_tokens = combined_mask.sum(dim=1)

                # 6. Calculate per-sample accuracy (correctness target for confidence)
                sample_correctness = torch.zeros_like(confidence, dtype=torch.float)
                valid_mask_sum = num_valid_tokens > 0
                sample_correctness[valid_mask_sum] = (
                    num_correct_tokens[valid_mask_sum].float() / (num_valid_tokens[valid_mask_sum].float() + 1e-9)
                )
                # --- End Corrected Calculation ---

            content_match = torch.all((predictions == target_labels) | (target_labels == pad_token_id), dim=1)

            pred_has_eos = (predictions == eos_token_id).any(dim=1)
            label_has_eos = (target_labels == eos_token_id).any(dim=1)

            pred_eos_pos = torch.argmax((predictions == eos_token_id).float(), dim=1)
            label_eos_pos = torch.argmax((target_labels == eos_token_id).float(), dim=1)

            pred_last_non_pad = torch.sum((predictions != pad_token_id).float(), dim=1) - 1
            label_last_non_pad = torch.sum((target_labels != pad_token_id).float(), dim=1) - 1
            pred_last_non_pad = torch.maximum(pred_last_non_pad, torch.zeros_like(pred_last_non_pad))
            label_last_non_pad = torch.maximum(label_last_non_pad, torch.zeros_like(label_last_non_pad))

            pred_end_pos = torch.where(pred_has_eos, pred_eos_pos, pred_last_non_pad)
            label_end_pos = torch.where(label_has_eos, label_eos_pos, label_last_non_pad)

            end_pos_match = pred_end_pos == label_end_pos

            eos_consistency = (pred_has_eos == label_has_eos)

            sequence_exact_match = content_match & end_pos_match & eos_consistency
            # --- End Sequence-level Exact Match Calculation ---

            # 4. Use sequence-level exact match as confidence target
            # 混合损失
            token_level_target = num_correct_tokens.float() / (num_valid_tokens.float() + 1e-9)
            sequence_level_target = sequence_exact_match.float()
            
            alpha = min(0.96, (epoch + 1) / args.num_epochs)
            combined_target = (1-alpha) * token_level_target + alpha * sequence_level_target
            
            conf_loss = conf_criterion(confidence, combined_target)

            # Combined loss (weighted sum)
            loss = gen_loss + args.confidence_weight * conf_loss

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            if accelerator.is_main_process:  # 只在主进程更新进度条
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        accelerator.wait_for_everyone()
        avg_epoch_loss = epoch_loss / len(train_loader)

        # --- Calibration Step ---
        after_demanded_epoch = (epoch + 1) >= args.when_to_calibrate
        calibrated_this_epoch = (
            after_demanded_epoch
            and (epoch + 1) % args.calibration_interval == 0
        )

        if calibrated_this_epoch:
            accelerator.print(f"Running confidence calibration (epoch {epoch + 1})...")
            # Train calibrator: returns calibrator object on main process, None otherwise
            # Threshold is now set internally by train_calibrator via broadcast
            new_calibrator_obj = train_calibrator(model, val_loader, tokenizer, accelerator, args)
            ever_calibrated = True
            if accelerator.is_main_process:
                last_calibrator = new_calibrator_obj # Store the latest calibrator
        else:
            accelerator.print(
                f"Skipping calibration for epoch {epoch + 1} (interval: {args.calibration_interval})"
            )
        # Ensure all processes have finished potential calibration steps and threshold updates
        accelerator.wait_for_everyone()


        # --- Validation Step ---
        accelerator.print(f"Epoch {epoch + 1} completed. Running evaluation...")
        # Pass the most recent calibrator (only exists on main process) to evaluate_model
        # Evaluate_model logic will handle calibrator being None on non-main processes
        metrics = evaluate_model(
            model=model,
            dataloader=val_loader, # Already prepared
            tokenizer=tokenizer,
            accelerator=accelerator,
            stage="val",
            calibrator=last_calibrator if accelerator.is_main_process else None, # Pass calibrator
            max_length=args.max_seq_length,
            num_beams=args.num_beams,
            max_resolve_len=args.max_resolve_len,
        )

        # Log validation metrics
        if accelerator.is_main_process:
            for key, value in metrics.items():
                accelerator.print(f"Val {key}: {value:.4f}")

        # --- Save Best Model State ---
        if accelerator.is_main_process:
            # Check F1 score instead of precision
            current_f1 = metrics.get("f1", 0.0)
            current_precision = metrics.get("precision", 0.0)
            not_converged = not ever_calibrated and current_f1 > best_f1
            better_precision = ever_calibrated and current_precision > best_precision
            if not_converged or better_precision:
                # Update best_f1 and best_precision
                best_f1 = current_f1
                best_precision = current_precision
                if not_converged:
                    accelerator.print(f"New best F1 score: {best_f1:.4f}. Saving state...")
                else:
                    accelerator.print(f"New best precision: {best_precision:.4f}. Saving state...")

                # Save accelerator state (model, optimizer, scheduler)
                accelerator.save_state(best_model_state_path)
                accelerator.print(f"Saved model state to {best_model_state_path}")

                # Save the calibrator that was used for this validation epoch
                if last_calibrator:
                    torch.save(last_calibrator, best_calibrator_path)
                    accelerator.print(f"Saved corresponding calibrator to {best_calibrator_path}")
                elif os.path.exists(best_calibrator_path):
                     # Remove old calibrator if no new one was trained this epoch but state is best
                     os.remove(best_calibrator_path)
                     accelerator.print(f"Removed stale calibrator from {best_calibrator_path}")


                # Save model configuration details
                unwrapped_model = accelerator.unwrap_model(model) # Get unwrapped model for config
                config_save_path = os.path.join(best_model_state_path, "config.txt")
                with open(config_save_path, "w") as f:
                    f.write(f"use_features: {args.use_features}\n")
                    f.write(f"feature_size: {args.feature_size}\n")
                    # Get the threshold *from the model state* which was saved by accelerator
                    f.write(
                        f"confidence_threshold: {unwrapped_model.confidence_threshold}\n"
                    )
                accelerator.print(f"Saved config to {config_save_path}")
                # No need to print this again: accelerator.print(f"Saved best model state to {best_model_state_path}")

        # Log epoch metrics to file
        if accelerator.is_main_process:
            current_threshold = accelerator.unwrap_model(model).confidence_threshold # Get current threshold
            metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in metrics.items())
            log_line = (
                f"Epoch {epoch + 1}:\t"
                f"Train Loss={avg_epoch_loss:.4f},\t"
                f"Val Metrics={{ {metrics_str} }},\t"
                f"Confidence Threshold={current_threshold:.4f},\t"
                f"Calibrated={calibrated_this_epoch}\n"
            )
            with open(train_log_path, "a") as f:
                f.write(log_line)

    # --- Final Evaluation on Test Set using Best Model State ---
    accelerator.wait_for_everyone()
    accelerator.print("***** Final evaluation on test set *****")

    best_calibrator_for_test = None # Initialize calibrator for test
    if os.path.exists(best_model_state_path):
        accelerator.print(f"Loading best model state from {best_model_state_path}...")
        accelerator.load_state(best_model_state_path)
        accelerator.print("Best model state loaded successfully.")

        # Load the corresponding calibrator ON MAIN PROCESS ONLY
        if accelerator.is_main_process:
            if os.path.exists(best_calibrator_path):
                try:
                    # Load the calibrator onto the CPU explicitly
                    best_calibrator_for_test = torch.load(best_calibrator_path, map_location=torch.device('cpu'))
                    accelerator.print(f"Loaded best calibrator from {best_calibrator_path}")
                except Exception as e:
                    accelerator.print(f"Warning: Failed to load best calibrator from {best_calibrator_path}: {e}")
            else:
                 accelerator.print(f"Warning: Best model state found, but no corresponding calibrator file at {best_calibrator_path}")

    else:
        accelerator.print("No best model state found. Evaluating using the final model state.")
        # Use the last calibrator trained during the run if evaluating final state
        if accelerator.is_main_process:
             best_calibrator_for_test = last_calibrator
             if best_calibrator_for_test:
                  accelerator.print("Using last trained calibrator for final model evaluation.")
             else:
                  accelerator.print("Warning: No best model state and no calibrator trained. Test evaluation will use raw scores.")


    # Prepare the test_loader *after* potentially loading the best state
    test_loader = accelerator.prepare(test_loader)

    # Run final evaluation, passing the loaded best calibrator
    # Only the main process needs/has the calibrator object
    final_metrics = evaluate_model(
        model=model, # Use the potentially loaded best model
        dataloader=test_loader,
        tokenizer=tokenizer,
        accelerator=accelerator,
        stage="test",
        calibrator=best_calibrator_for_test if accelerator.is_main_process else None, # Pass loaded calibrator
        max_length=args.max_seq_length,
        num_beams=args.num_beams,
        max_resolve_len=args.max_resolve_len,
    )

    # Log final test metrics
    if accelerator.is_main_process:
        log_final_metrics = {}
        accelerator.print("--- Final Test Metrics (using best model) ---")
        for key, value in final_metrics.items():
            accelerator.print(f"Test {key}: {value:.4f}")
            log_final_metrics[f"test_{key}"] = value
        # accelerator.log(log_final_metrics, step=args.num_epochs) # Log final metrics

    # Record end time and calculate duration
    end_time = time.time()
    total_duration = end_time - start_time

    # Log final test metrics and time to files
    if accelerator.is_main_process:
        test_log_line = f"Test Metrics (Best Model): {{{', '.join(f'{k}={v:.4f}' for k, v in final_metrics.items())}}}\n"
        with open(test_log_path, "a") as f:
            f.write(test_log_line) # Append to test log

        time_log_line = f"Total Training Time: {total_duration:.2f} seconds\n"
        with open(time_log_path, "a") as f:
            f.write(time_log_line) # Append to time log


    accelerator.end_training() # Clean up accelerator resources
    accelerator.print("Training completed!")


def train_calibrator(model: HesitateT5, val_loader: DataLoader, tokenizer: AutoTokenizer, accelerator: Accelerator, args: argparse.Namespace):
    """
    Train the confidence calibrator on the main process using gathered validation data.
    Sets the threshold on the model across all processes via broadcast.
    Returns the calibrator object on the main process, None otherwise.
    """
    model.eval()
    raw_confidences = []
    exact_matches_for_calib = []
    disable_progress_bar = not accelerator.is_main_process
    pad_token_id = tokenizer.pad_token_id

    # --- Collect Data ---
    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Collecting calibration data", disable=disable_progress_bar
        ):
            batch["stage"] = "val"
            outputs = model(**batch)
            confidence = outputs["confidence"]
            predictions = outputs["predictions"]
            labels = batch["labels"]

            # --- Vectorized Exact Match Calculation (for calibrator target) ---
            # Ensure predictions and labels are of the same length
            batch_size, seq_len_pred = predictions.shape
            _, seq_len_label = labels.shape
            
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
            
            content_match = torch.all((predictions == labels) | (labels == pad_token_id), dim=1)

            eos_token_id = tokenizer.eos_token_id
            pred_has_eos = (predictions == eos_token_id).any(dim=1)
            label_has_eos = (labels == eos_token_id).any(dim=1)

            pred_eos_pos = torch.argmax((predictions == eos_token_id).float(), dim=1)
            label_eos_pos = torch.argmax((labels == eos_token_id).float(), dim=1)

            pred_last_non_pad = torch.sum((predictions != pad_token_id).float(), dim=1) - 1
            label_last_non_pad = torch.sum((labels != pad_token_id).float(), dim=1) - 1
            # 确保不是负数 (如果全是pad)
            pred_last_non_pad = torch.maximum(pred_last_non_pad, torch.zeros_like(pred_last_non_pad))
            label_last_non_pad = torch.maximum(label_last_non_pad, torch.zeros_like(label_last_non_pad))

            pred_end_pos = torch.where(pred_has_eos, pred_eos_pos, pred_last_non_pad)
            label_end_pos = torch.where(label_has_eos, label_eos_pos, label_last_non_pad)

            end_pos_match = pred_end_pos == label_end_pos

            eos_consistency = (pred_has_eos == label_has_eos)

            exact_match = content_match & end_pos_match & eos_consistency

            raw_confidences.extend(confidence.cpu().tolist())
            exact_matches_for_calib.extend(exact_match.cpu().tolist()) # Use exact match as target

    # --- Gather Data Across Processes ---
    gathered_confidences_list = accelerator.gather_for_metrics(raw_confidences)
    gathered_exact_matches_list = accelerator.gather_for_metrics(exact_matches_for_calib)

    calibrator = None
    new_threshold = None
    unwrapped_model = accelerator.unwrap_model(model) # Unwrap once here

    # --- Calculate Calibrator and Threshold on Main Process Only ---
    if accelerator.is_main_process:
        # Directly use the gathered lists, NO flattening needed
        flat_confidences = gathered_confidences_list
        flat_exact_matches = gathered_exact_matches_list

        current_threshold = unwrapped_model.confidence_threshold # Get current as fallback

        if len(flat_confidences) < 10 or len(set(flat_confidences)) < 2:
            accelerator.print(
                f"Warning: Too few samples ({len(flat_confidences)}) or unique confidence values ({len(set(flat_confidences))}) for calibration. Keeping current threshold {current_threshold:.4f}."
            )
            new_threshold = current_threshold # Keep old threshold
            calibrator = None # Ensure no old calibrator is returned
        else:
            try:
                # Fit calibrator using exact match as the correctness target
                calibrator = ConfidenceCalibrator(strategy=args.calibration_strategy, initial_threshold=current_threshold)
                # Pass boolean list directly to fit
                calibrator.fit(flat_confidences, flat_exact_matches)
                calculated_t = calibrator.get_threshold() # get_threshold finds based on F0.5 of calibrated scores

                if calculated_t is None: # Handle calibrator potentially returning None
                    accelerator.print(f"Warning: Calibrator returned None threshold. Keeping current threshold {current_threshold:.4f}.")
                    new_threshold = current_threshold
                    # Keep the fitted calibrator object even if threshold is None, might be useful later? Or set to None? Let's set to None for consistency.
                    calibrator = None
                else:
                    new_threshold = calculated_t
                    accelerator.print(f"Calibration complete. Proposed new threshold: {new_threshold:.4f}")

            except Exception as e:
                 accelerator.print(f"Error during calibration fitting or threshold calculation: {e}. Keeping current threshold {current_threshold:.4f}.")
                 new_threshold = current_threshold # Fallback on error
                 calibrator = None # Ensure no failed calibrator is returned

    # Wait for main process to finish calculation before broadcasting threshold
    accelerator.wait_for_everyone()

    # --- Broadcast Threshold ---
    # Prepare the threshold for broadcast
    if accelerator.is_main_process:
        # Use the calculated new_threshold if valid, otherwise fallback to current
        threshold_val = float(new_threshold if new_threshold is not None else current_threshold)
        threshold_tensor = torch.tensor(threshold_val, device=accelerator.device)
    else:
        threshold_tensor = torch.empty(1, device=accelerator.device) # Placeholder on other processes

    # Broadcast the tensor from process 0 using torch distributed (more robust than utils)
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(threshold_tensor, src=0)
        # All processes receive the threshold in their tensor
        new_threshold = threshold_tensor.item() # Extract float value
    elif accelerator.is_main_process:
         # Single process: new_threshold already has the value (or fallback)
         new_threshold = threshold_val


    # --- Set Threshold on All Processes ---
    # Ensure new_threshold is a float before setting
    if new_threshold is not None:
        unwrapped_model.set_confidence_threshold(float(new_threshold))
        # Log on all processes to confirm the *applied* threshold
        current_model_threshold = unwrapped_model.confidence_threshold # Read back
        accelerator.print(
            f"Process {accelerator.process_index}: Confidence threshold set to {current_model_threshold:.4f}."
        )
    else:
         # Should not happen with the fallback logic, but log just in case
         accelerator.print(f"Process {accelerator.process_index}: Threshold remained None after broadcast/calculation.")


    # Return the calibrator object (only exists on main process)
    return calibrator


def main():
    parser = argparse.ArgumentParser(description="Train merge-hesitate model")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./codet5-small",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="./tokenizer",
        help="Path to pretrained tokenizer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_resolve_len",
        type=int,
        default=256,
        help="Maximum length for resolving text",
    )
    parser.add_argument(
        "--feature_size", type=int, default=12, help="Size of feature space to reserve"
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Whether to use extracted features (default: False)",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=36, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=6,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="OUTPUT",
        help="Output directory for logs, checkpoints, etc.", # Clarified help text
    )

    # Generation arguments
    parser.add_argument(
        "--num_beams", type=int, default=3, help="Number of beams for beam search"
    )

    # Calibration arguments
    parser.add_argument(
        "--calibration_strategy",
        type=str,
        default="temperature",
        choices=["temperature", "isotonic"],
        help="Calibration strategy (temperature or isotonic)",
    )
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=1,
        help="Weight for confidence loss",
    )
    parser.add_argument(
        "--initial_threshold",
        type=float,
        default=0.8,
        help="Initial confidence threshold",
    )
    parser.add_argument(
        "--calibration_interval",
        type=int,
        default=5,
        help="Run confidence calibration every N epochs after the start epoch.",
    )

    parser.add_argument(
        "--when_to_calibrate",
        type=int,
        default=50,
        help="Run confidence calibration after N epochs.",
    )


    args = parser.parse_args()

    # Validate and create output directory (Accelerator might handle this with project_dir)
    # os.makedirs(args.output_dir, exist_ok=True) # Keep for safety or rely on Accelerator

    # Start training
    train(args)


if __name__ == "__main__":
    main()
