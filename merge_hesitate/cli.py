#!/usr/bin/env python3
"""
Command-line interface for merge-hesitate
"""

import argparse
import os
import sys
import logging
from transformers import AutoTokenizer
import torch

from merge_hesitate.model import HesitateT5
from merge_hesitate.features import FeatureExtractor, prepare_input_with_features

logger = logging.getLogger(__name__)


def setup_logger():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_model(output_dir, tokenizer=None):
    """Load the best trained model and calibrator from the training output directory."""

    best_model_path = os.path.join(output_dir, "best_model")
    calibrator_path = os.path.join(output_dir, "calibrator", "calibrator.pt")
    config_path = os.path.join(best_model_path, "config.txt")
    confidence_head_path = os.path.join(best_model_path, "confidence_head.pt")
    feature_projector_path = os.path.join(best_model_path, "feature_projector.pt")

    if not os.path.exists(best_model_path):
        logger.error(f"Best model directory not found: {best_model_path}")
        sys.exit(1)

    # --- 1. Load Configuration ---
    use_features = False
    feature_size = 12  # Default from HesitateT5 init
    initial_threshold = 0.8  # Default from HesitateT5 init

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = {}
                for line in f:
                    key, value = line.strip().split(": ")
                    config_data[key] = value

                use_features = (
                    config_data.get("use_features", "false").lower() == "true"
                )
                feature_size = int(config_data.get("feature_size", feature_size))
                initial_threshold = float(
                    config_data.get("confidence_threshold", initial_threshold)
                )
            logger.info(
                f"Loaded config from {config_path}: use_features={use_features}, feature_size={feature_size}, initial_threshold={initial_threshold:.4f}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to parse {config_path}: {e}. Using default config values."
            )
    else:
        logger.warning(f"{config_path} not found. Using default config values.")

    # --- 2. Initialize Model ---
    try:
        # Load the base T5 model using the directory path
        model = HesitateT5(
            model_name_or_path=best_model_path,  # Pass the directory to from_pretrained
            tokenizer=tokenizer,
            feature_size=feature_size,
            use_features=use_features,
            confidence_threshold=initial_threshold,  # Set initial threshold from config/default
        )
        logger.info(f"Initialized base model from {best_model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize base model from {best_model_path}: {e}")
        sys.exit(1)

    # --- 3. Load Confidence Head ---
    if os.path.exists(confidence_head_path):
        try:
            model.confidence_head.load_state_dict(
                torch.load(confidence_head_path, map_location="cpu")
            )
            logger.info(f"Loaded confidence head from {confidence_head_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load confidence head state dict from {confidence_head_path}: {e}"
            )
    else:
        logger.warning(
            f"Confidence head state dict not found at {confidence_head_path}. Using initialized weights."
        )

    # --- 4. Load Feature Projector (if needed) ---
    if use_features:
        if os.path.exists(feature_projector_path):
            try:
                model.feature_projector.load_state_dict(
                    torch.load(feature_projector_path, map_location="cpu")
                )
                logger.info(f"Loaded feature projector from {feature_projector_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load feature projector state dict from {feature_projector_path}: {e}"
                )
        else:
            logger.warning(
                f"Feature projector state dict not found at {feature_projector_path}. Using initialized weights."
            )

    # --- 5. Load Calibrator and Set Final Threshold ---
    if os.path.exists(calibrator_path):
        try:
            calibrator = torch.load(calibrator_path, map_location="cpu")
            final_threshold = calibrator.get_threshold()
            model.set_confidence_threshold(final_threshold)
            logger.info(
                f"Loaded calibrator from {calibrator_path} and set final threshold to {final_threshold:.4f}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load calibrator from {calibrator_path}: {e}. Model will use threshold {model.confidence_threshold:.4f} (from config/default). "
            )
    else:
        logger.info(
            f"Calibrator not found at {calibrator_path}. Model will use threshold {model.confidence_threshold:.4f} (from config/default)."
        )

    return model


def resolve_conflict(
    conflict_text,
    model,
    tokenizer,
    feature_extractor,
    device="cuda",
    max_length=512,
    num_beams=4,
    max_resolve_len=256,
):
    """Resolve a single merge conflict"""
    # Prepare inputs with features
    inputs, features = prepare_input_with_features(
        tokenizer, conflict_text, feature_extractor
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate solution
    with torch.no_grad():
        outputs, confidence = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            max_resolve_len=max_resolve_len,
        )

    # Decode solution
    solution_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    confidence_value = confidence[0].item()

    # Return solution only if confident enough
    if confidence_value >= model.confidence_threshold:
        return solution_text, confidence_value
    else:
        # Return None for solution if confidence is too low
        return None, confidence_value


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Resolve git merge conflicts")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the training output directory containing 'best_model' and 'calibrator' subdirectories",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        default="./tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--input_file", type=str, help="File containing merge conflict to resolve"
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file for resolved conflict"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, help="Override confidence threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for generation",
    )
    parser.add_argument(
        "--num_beams", type=int, default=3, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--max_resolve_len",
        type=int,
        default=256,
        help="Maximum length for resolving text",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logger()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = load_model(args.output_dir, tokenizer).to(args.device)
    feature_extractor = FeatureExtractor()

    # Set custom threshold if provided
    if args.confidence_threshold:
        model.set_confidence_threshold(args.confidence_threshold)
        logger.info(f"Confidence threshold set to {args.confidence_threshold}")

    # Read conflict from file or stdin
    if args.input_file:
        with open(args.input_file, "r") as f:
            conflict_text = f.read()
    else:
        conflict_text = sys.stdin.read()

    # Resolve conflict
    solution, confidence = resolve_conflict(
        conflict_text,
        model,
        tokenizer,
        feature_extractor,
        args.device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        max_resolve_len=args.max_resolve_len,
    )

    # Print results based on whether a confident solution was found
    if solution is not None:
        logger.info(f"Solution found with confidence {confidence:.4f}")

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(solution)
            logger.info(f"Resolved conflict saved to {args.output_file}")
        else:
            print(solution)
    else:
        logger.info(
            f"No confident solution found (confidence: {confidence:.4f}). Threshold: {model.confidence_threshold:.4f}"
        )
        # Exit with non-zero status code to indicate failure
        sys.exit(1)


if __name__ == "__main__":
    main()
