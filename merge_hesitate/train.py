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

from .model import HesitateT5
from .data import create_dataloaders
from .confidence import ConfidenceCalibrator
from .evaluate import evaluate_model

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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
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
    special_tokens = ["<lbra>", "<rbra>"]

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
        max_resolve_length=args.max_resolve_length,
        feature_size=args.feature_size,
        use_features=args.use_features,
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
    best_accuracy = 0.0

    # 创建tqdm进度条，只在主进程显示
    disable_progress_bar = not accelerator.is_main_process

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
            outputs = model(**batch)

            # Get loss and other metrics
            gen_loss = outputs["loss"] / outputs["valid_tokens"]
            # size: [batch_size]
            confidence = outputs["confidence"]

            # Get model predictions
            with torch.no_grad():
                # Get predictions from logits [batch_size, seq_len, vocab_size] -> [batch_size, seq_len]
                predictions = torch.argmax(outputs["logits"], dim=-1)

                # Create shifted labels to match prediction positions (like in training)
                # batch_size, seq_len -> batch_size, seq_len + 1
                shifted_labels = batch["labels"].clone()
                shifted_labels = torch.cat(
                    [
                        shifted_labels,
                        torch.ones(
                            (shifted_labels.size(0), 1),
                            dtype=torch.long,
                            device=shifted_labels.device,
                        )
                        * tokenizer.pad_token_id,
                    ],
                    dim=-1,
                )
                # batch_size, seq_len + 1 -> batch_size, seq_len
                shifted_labels = shifted_labels[:, 1:]

                # Create mask for valid positions (non-padding), size: [batch_size, seq_len]
                mask = shifted_labels != tokenizer.pad_token_id

                # Compare predictions with ground truth where mask is True
                # For each sample, calculate if predictions match targets (averaged across sequence)
                sample_correct = torch.zeros_like(confidence)

                for i in range(predictions.size(0)):
                    # Get valid predictions and labels for this sample
                    valid_preds = predictions[i][mask[i]]
                    valid_labels = shifted_labels[i][mask[i]]

                    if len(valid_preds) > 0:
                        # Calculate accuracy for this sample (between 0 and 1)
                        accuracy = (valid_preds == valid_labels).float().mean()
                        sample_correct[i] = accuracy

            # Use correctness as target for confidence
            conf_loss = conf_criterion(confidence, sample_correct)

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

        # Evaluate after each epoch
        accelerator.print(f"Epoch {epoch + 1} completed. Running evaluation...")
        val_loader = accelerator.prepare(val_loader)
        metrics = evaluate_model(
            model,
            val_loader,
            tokenizer,
            accelerator,
            max_length=args.max_seq_length,
            num_beams=args.num_beams,
            max_resolve_len=args.max_resolve_len,
        )

        # Log metrics (只在主进程显示详细指标)
        if accelerator.is_main_process:
            for key, value in metrics.items():
                accelerator.print(f"{key}: {value:.4f}")

        # Save best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]

            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(output_dir, exist_ok=True)

                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.generator.save_pretrained(output_dir)

                # Save confidence head and feature projector separately
                torch.save(
                    unwrapped_model.confidence_head.state_dict(),
                    os.path.join(output_dir, "confidence_head.pt"),
                )
                torch.save(
                    unwrapped_model.feature_projector.state_dict(),
                    os.path.join(output_dir, "feature_projector.pt"),
                )

                # Save model configuration details
                with open(os.path.join(output_dir, "config.txt"), "w") as f:
                    f.write(f"use_features: {args.use_features}\n")
                    f.write(f"feature_size: {args.feature_size}\n")
                    f.write(
                        f"confidence_threshold: {unwrapped_model.confidence_threshold}\n"
                    )

                accelerator.print(f"Saved best model to {output_dir}")

        # Train confidence calibrator periodically after the initial epochs
        if (
            epoch >= args.num_epochs // 2
            and (epoch + 1) % args.calibration_interval == 0
        ):
            accelerator.print(f"Running confidence calibration (epoch {epoch + 1})...")
            # Ensure val_loader is prepared if it wasn't already in this scope
            prepared_val_loader = accelerator.prepare(val_loader)
            train_calibrator(model, prepared_val_loader, tokenizer, accelerator, args)
        else:
            accelerator.print(
                f"Skipping calibration for epoch {epoch + 1} (interval: {args.calibration_interval})"
            )

    # Final evaluation on test set
    accelerator.wait_for_everyone()
    accelerator.print("***** Final evaluation on test set *****")
    test_loader = accelerator.prepare(test_loader)
    metrics = evaluate_model(
        model,
        test_loader,
        tokenizer,
        accelerator,
        max_length=args.max_seq_length,
        num_beams=args.num_beams,
        max_resolve_len=args.max_resolve_len,
    )

    # Log metrics
    if accelerator.is_main_process:
        for key, value in metrics.items():
            accelerator.print(f"{key}: {value:.4f}")

    accelerator.print("Training completed!")


def train_calibrator(model, val_loader, tokenizer, accelerator, args):
    """Train the confidence calibrator using validation data."""
    model.eval()
    raw_confidences = []
    correctness = []

    # 只在主进程显示进度条
    disable_progress_bar = not accelerator.is_main_process

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Collecting calibration data", disable=disable_progress_bar
        ):
            # Set evaluation stage
            batch["stage"] = "eval"

            # Add generation parameters
            batch["max_resolve_len"] = args.max_resolve_len
            batch["num_beams"] = args.num_beams
            batch["max_length"] = args.max_seq_length

            # Get model predictions
            generated, confidence = model.generate(**batch)

            # Check if predictions match targets
            correct = []
            for i in range(len(generated)):
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = tokenizer.decode(
                    batch["labels"][i], skip_special_tokens=True
                )
                # 将生成的文本和目标文本截断为256个词语
                gen_text_words = gen_text.split()[: args.max_resolve_len]
                target_text_words = target_text.split()[: args.max_resolve_len]
                gen_text = " ".join(gen_text_words)
                target_text = " ".join(target_text_words)

                # 只有当置信度大于阈值时，才判断预测是否正确
                # 否则标记为不正确，因为模型"拒绝回答"
                if confidence[i] >= model.confidence_threshold:
                    correct.append(gen_text == target_text)
                else:
                    correct.append(False)  # 低置信度视为错误预测

            # Collect data for calibration
            raw_confidences.extend(confidence.cpu().tolist())
            correctness.extend(correct)

    # 收集所有进程的数据
    if accelerator.num_processes > 1:
        raw_confidences = accelerator.gather_for_metrics(raw_confidences)
        correctness = accelerator.gather_for_metrics(correctness)

    # 主进程计算校准器
    if accelerator.is_main_process:
        # 检查收集的数据是否足够
        if len(raw_confidences) < 10:
            accelerator.print(
                "Warning: Too few samples for calibration, keeping previous threshold"
            )
            # Get the unwrapped model and access its current threshold
            unwrapped_model = accelerator.unwrap_model(model)
            new_threshold = unwrapped_model.confidence_threshold
        else:
            # Train calibrator
            calibrator = ConfidenceCalibrator(strategy=args.calibration_strategy)
            calibrator.fit(raw_confidences, correctness)
            new_threshold = calibrator.get_threshold()
            accelerator.print(f"Calibrated confidence threshold: {new_threshold:.4f}")

            # Save calibrator
            output_dir = os.path.join(args.output_dir, "calibrator")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(calibrator, os.path.join(output_dir, "calibrator.pt"))
    else:
        # Handle the case where calibration wasn't run on the main process
        # We still need a value for new_threshold to avoid errors
        # Get the current threshold from the model as a default
        unwrapped_model = accelerator.unwrap_model(model)
        new_threshold = unwrapped_model.confidence_threshold

    # Broadcast the threshold determined by the main process (or the default if calibration failed/skipped)
    if accelerator.num_processes > 1:
        # Ensure new_threshold is a tensor for broadcasting if it's a Python float
        threshold_tensor = torch.tensor(new_threshold, device=accelerator.device)
        accelerator.broadcast(threshold_tensor, from_process=0)
        new_threshold = threshold_tensor.item()  # Convert back to float

    # Set the threshold on the model across all processes
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.set_confidence_threshold(new_threshold)
    accelerator.print(
        f"Confidence threshold set to {new_threshold:.4f} across all processes."
    )


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
        "--max_resolve_length",
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
        help="Output directory",
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
        "--max_resolve_len",
        type=int,
        default=256,
        help="Maximum length for resolving text",
    )
    # Add calibration interval argument
    parser.add_argument(
        "--calibration_interval",
        type=int,
        default=5,
        help="Run confidence calibration every N epochs after the halfway point.",
    )

    args = parser.parse_args()

    # Validate and create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
