#!/usr/bin/env python3
"""
Command-line interface for merge-hesitate
"""

import argparse
import os
import sys
import logging
from transformers import T5Tokenizer
import torch

from .model import HesitateT5
from .features import FeatureExtractor, prepare_input_with_features

logger = logging.getLogger(__name__)


def setup_logger():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_model(model_path):
    """Load a trained model from disk"""
    # Load generator
    generator = HesitateT5(model_name_or_path=model_path)

    # Load confidence head
    confidence_path = os.path.join(model_path, "confidence_head.pt")
    if os.path.exists(confidence_path):
        generator.confidence_head.load_state_dict(torch.load(confidence_path))

    # Load calibrator
    calibrator_path = os.path.join(
        os.path.dirname(model_path), "calibrator", "calibrator.pt"
    )
    if os.path.exists(calibrator_path):
        calibrator = torch.load(calibrator_path)
        generator.set_confidence_threshold(calibrator.get_threshold())

    return generator


def resolve_conflict(conflict_text, model, tokenizer, feature_extractor, device="cuda"):
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
            max_length=512,
            num_beams=4,
        )

    # Decode solution
    solution_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence_value = confidence[0].item()

    # Return solution only if confident enough
    if confidence_value >= model.confidence_threshold:
        return solution_text, confidence_value
    else:
        return None, confidence_value


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Resolve git merge conflicts")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
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

    args = parser.parse_args()

    # Set up logging
    setup_logger()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path).to(args.device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
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
        conflict_text, model, tokenizer, feature_extractor, args.device
    )

    # Print results
    if solution:
        logger.info(f"Solution found with confidence {confidence:.4f}")

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(solution)
        else:
            print(solution)
    else:
        logger.info(f"No confident solution found (confidence: {confidence:.4f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
