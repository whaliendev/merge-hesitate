import torch
import torch.nn as nn
import os
import logging
import argparse
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json # For reading config potentially

# Assuming these modules are accessible from the execution environment
from merge_hesitate.model import HesitateT5
from merge_hesitate.data import create_dataloaders
from merge_hesitate.confidence import ConfidenceCalibrator
from merge_hesitate.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def load_config_from_state_path(state_path):
    """Loads configuration parameters from config.txt in the state path."""
    config_path = os.path.join(state_path, "config.txt")
    config = {}
    defaults = { # Provide defaults in case the config file is missing fields
        'use_features': False,
        'feature_size': 12,
        'confidence_threshold': 0.5 # Default threshold if not found
    }
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default values: {defaults}")
        return defaults

    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == 'use_features':
                    config[key] = value.lower() == 'true'
                elif key == 'feature_size':
                    config[key] = int(value)
                elif key == 'confidence_threshold':
                    config[key] = float(value)
                # Add other config parameters if needed
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}. Using default values: {defaults}")
        return defaults

    # Fill missing keys with defaults
    for key, default_value in defaults.items():
        if key not in config:
            logger.warning(f"Config key '{key}' not found in {config_path}. Using default: {default_value}")
            config[key] = default_value

    logger.info(f"Loaded config from {config_path}: {config}")
    return config


def test(args):
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # Might be needed if model parts aren't used in test
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Starting test run with Accelerator state: {accelerator.state}")

    # --- Load Configuration from Saved State ---
    if not os.path.exists(args.model_state_path):
        logger.error(f"Model state path not found: {args.model_state_path}")
        return
    loaded_config = load_config_from_state_path(args.model_state_path)
    use_features = loaded_config['use_features']
    feature_size = loaded_config['feature_size']
    loaded_threshold = loaded_config['confidence_threshold'] # Threshold from training

    # --- Load Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        special_tokens = ["<lbra>", "<rbra>"]
        num_added = tokenizer.add_tokens(special_tokens)
        if num_added != len(special_tokens):
             logger.warning(f"Added {num_added}/{len(special_tokens)} special tokens. Some might have existed.")
        logger.info("Tokenizer loaded and special tokens checked.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {args.tokenizer_name_or_path}: {e}")
        return

    # --- Create Test Dataloader ---
    # We only need the test loader for evaluation
    try:
        _, _, test_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            max_resolve_length=args.max_resolve_len,
            feature_size=feature_size, # Use loaded value
            use_features=use_features, # Use loaded value
            accelerator=accelerator, # Pass accelerator for distributed sampling
        )
        logger.info("Test dataloader created.")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        return

    # --- Initialize Model Structure ---
    try:
        model_structure = HesitateT5(
            model_name_or_path=args.model_name_or_path, # Base structure path
            tokenizer=tokenizer, # Pass tokenizer for potential resizing
            confidence_threshold=0.5, # Initial threshold, will be overwritten
            feature_size=feature_size, # Use loaded value
            use_features=use_features, # Use loaded value
        )
        logger.info("HesitateT5 model structure initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize HesitateT5 model structure: {e}")
        return

    # --- Prepare Model and Dataloader ---
    # Important: Prepare *before* loading state
    try:
        model, test_loader = accelerator.prepare(model_structure, test_loader)
        logger.info("Model and test loader prepared with Accelerator.")
    except Exception as e:
        logger.error(f"Failed to prepare model/dataloader with Accelerator: {e}")
        return


    # --- Load Model State ---
    try:
        accelerator.load_state(args.model_state_path)
        logger.info(f"Successfully loaded model state from {args.model_state_path}")

        # --- Set Correct Threshold AFTER Loading State ---
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.set_confidence_threshold(loaded_threshold)
        logger.info(f"Set model confidence threshold to loaded value: {unwrapped_model.confidence_threshold:.4f}")

    except Exception as e:
        logger.error(f"Failed to load model state from {args.model_state_path}: {e}")
        return

    # --- Load Calibrator ---
    calibrator_path = os.path.join(args.model_state_path, "calibrator.pt")
    best_calibrator_for_test = None
    if accelerator.is_main_process: # Only main process loads/needs the calibrator object
        if os.path.exists(calibrator_path):
            try:
                # Load the calibrator onto the CPU explicitly
                best_calibrator_for_test = torch.load(calibrator_path, map_location=torch.device('cpu'), weights_only=False)
                # Keep internal tensors on CPU as well
                if hasattr(best_calibrator_for_test, 'temperature') and isinstance(best_calibrator_for_test.temperature, torch.Tensor):
                     best_calibrator_for_test.temperature = best_calibrator_for_test.temperature.to(torch.device('cpu'))
                logger.info(f"Loaded best calibrator from {calibrator_path} onto CPU.")
            except Exception as e:
                logger.warning(f"Failed to load best calibrator from {calibrator_path}: {e}. Proceeding without calibration.")
        else:
             logger.warning(f"Best model state loaded, but no corresponding calibrator file found at {calibrator_path}. Proceeding without calibration.")


    # --- Run Final Evaluation ---
    accelerator.print("***** Running final evaluation on test set *****")
    try:
        final_metrics = evaluate_model(
            model=model, # Pass the prepared and loaded model
            dataloader=test_loader,
            tokenizer=tokenizer,
            accelerator=accelerator,
            stage="test",
            calibrator=best_calibrator_for_test if accelerator.is_main_process else None, # Pass loaded calibrator (main proc only)
            max_length=args.max_seq_length, # Ensure this aligns with dataloader/training
            num_beams=args.num_beams,
            max_resolve_len=args.max_resolve_len,
        )
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}", exc_info=True)
        final_metrics = {} # Report empty metrics on error

    # Log final test metrics
    if accelerator.is_main_process:
        if final_metrics:
            accelerator.print("--- Final Test Metrics (Loaded Best Model) ---")
            metrics_str = ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in final_metrics.items())
            accelerator.print(f"Test Metrics: {{ {metrics_str} }}")

            # Optionally save metrics to a file
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, "test_results.json")
                try:
                    with open(output_file, "w") as f:
                        json.dump(final_metrics, f, indent=4)
                    logger.info(f"Test metrics saved to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save test metrics to {output_file}: {e}")
        else:
            accelerator.print("Evaluation resulted in empty metrics, possibly due to an error.")

    accelerator.print("Testing script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone evaluation on the test set using the best saved model state.")

    # --- Required Paths ---
    parser.add_argument(
        "--model_state_path",
        type=str,
        required=True,
        help="Path to the directory containing the saved 'best_model_state' (output by train.py).",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="./tokenizer",
        help="Path to the pretrained tokenizer.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./codet5-small",
        help="Path to the base pretrained model used for structure initialization (e.g., './codet5-small').",
    )

    # --- Data Parameters ---
    parser.add_argument("--batch_size", type=int, default=36, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length (must match training).")
    parser.add_argument("--max_resolve_len", type=int, default=256, help="Maximum length for generated resolved text.")
    # Note: feature_size and use_features are loaded from config.txt

    # --- Evaluation Parameters ---
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search generation.")

    # --- Optional Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None, # Default to None, only save if specified
        help="Optional directory to save test results (test_results.json).",
    )


    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.model_state_path):
        print(f"Error: Provided model_state_path does not exist or is not a directory: {args.model_state_path}")
        exit(1)
    if not os.path.isdir(args.tokenizer_name_or_path):
         print(f"Error: Provided tokenizer_name_or_path does not exist or is not a directory: {args.tokenizer_name_or_path}")
         exit(1)
    if not os.path.isdir(args.model_name_or_path):
         print(f"Warning: Provided model_name_or_path does not exist or is not a directory: {args.model_name_or_path}. Make sure it points to the correct base model structure.")
         # Don't exit, maybe user knows it's just an identifier for transformers lib

    test(args) 