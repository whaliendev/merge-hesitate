import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, T5Tokenizer
import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator

from ..model import HesitateT5
from ..data import create_dataloaders
from ..features import FeatureExtractor
from ..evaluate import evaluate_model
from .model import MergeHesitateGAN
from .evaluate import evaluate_gan_model
from .utils import generate_samples_for_discriminator

logger = logging.getLogger(__name__)


def train_gan(args):
    """Train the GAN-based merge conflict resolution model"""
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

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # Create feature extractor
    feature_extractor = FeatureExtractor(max_seq_length=args.max_seq_length)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_dir,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        use_augmentation=args.use_augmentation,
    )

    # Initialize model
    # Phase 1: If we're starting from scratch, first train the generator
    if args.pretrained_generator_path is None:
        logger.info("No pretrained generator provided. Training from scratch.")
        generator = HesitateT5(model_name_or_path=args.model_name_or_path)

        if args.phase == "generator_pretraining":
            # Train just the generator first
            logger.info("Phase 1: Pretraining generator")
            generator = train_generator(
                generator, train_loader, val_loader, tokenizer, args, accelerator
            )
    else:
        # Load pretrained generator
        logger.info(
            f"Loading pretrained generator from {args.pretrained_generator_path}"
        )
        generator = HesitateT5(model_name_or_path=args.pretrained_generator_path)

    # Initialize GAN
    gan_model = MergeHesitateGAN(
        generator_path=args.model_name_or_path,
        discriminator_path=args.model_name_or_path,
        confidence_threshold=args.initial_threshold,
        pretrained_generator=generator,
    )

    # Phase 2: Train the discriminator
    if args.phase == "discriminator_training":
        logger.info("Phase 2: Training discriminator")
        gan_model = train_discriminator(
            gan_model, train_loader, val_loader, tokenizer, args, accelerator
        )

    # Phase 3: Adversarial training of the full GAN
    elif args.phase == "adversarial_training":
        logger.info("Phase 3: Adversarial training")
        gan_model = train_adversarial(
            gan_model, train_loader, val_loader, tokenizer, args, accelerator
        )

    # Final evaluation on test set
    logger.info("***** Final evaluation on test set *****")
    test_loader = accelerator.prepare(test_loader)
    metrics = evaluate_gan_model(gan_model, test_loader, tokenizer, accelerator)

    # Log metrics
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    logger.info("Training completed!")


def train_generator(model, train_loader, val_loader, tokenizer, args, accelerator):
    """Train only the generator model"""
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

    # Define loss functions
    conf_criterion = nn.BCELoss()

    # Training loop
    logger.info("***** Starting generator training *****")
    global_step = 0
    best_accuracy = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Generation loss (from T5)
            gen_loss = outputs.loss

            # Confidence loss (binary cross-entropy with correctness labels)
            if "is_correct" in batch:
                confidence = outputs.confidence
                conf_loss = conf_criterion(confidence, batch["is_correct"])

                # Combined loss (weighted sum)
                loss = gen_loss + args.confidence_weight * conf_loss
            else:
                loss = gen_loss

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
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        # Evaluate after each epoch
        logger.info(f"Epoch {epoch + 1} completed. Running evaluation...")
        metrics = evaluate_model(model, val_loader, tokenizer, accelerator)

        # Log metrics
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        # Save best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]

            # Save model
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_dir, "best_generator")
                os.makedirs(output_dir, exist_ok=True)

                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.generator.save_pretrained(output_dir)

                # Save confidence head separately
                torch.save(
                    unwrapped_model.confidence_head.state_dict(),
                    os.path.join(output_dir, "confidence_head.pt"),
                )

                logger.info(f"Saved best generator to {output_dir}")

    return accelerator.unwrap_model(model)


def train_discriminator(
    gan_model, train_loader, val_loader, tokenizer, args, accelerator
):
    """Train only the discriminator part of the GAN"""
    # Set models to appropriate training modes
    gan_model.train_discriminator_only()

    # Set up optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, gan_model.parameters()),
        lr=args.learning_rate_discriminator,
    )

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
    gan_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        gan_model, optimizer, train_loader, val_loader, scheduler
    )

    # Define loss function
    bce_loss = nn.BCELoss()

    logger.info("***** Starting discriminator training *****")
    global_step = 0
    best_score = 0.0

    # First, generate samples for discriminator training
    logger.info("Generating samples for discriminator training...")
    discriminator_samples = generate_samples_for_discriminator(
        gan_model,
        train_loader,
        tokenizer,
        accelerator,
        num_samples=min(10000, len(train_loader.dataset)),
    )

    # Training loop
    for epoch in range(args.num_epochs):
        gan_model.train_discriminator_only()
        epoch_loss = 0.0

        # Shuffle discriminator samples
        np.random.shuffle(discriminator_samples)

        progress_bar = tqdm(
            range(0, len(discriminator_samples), args.batch_size),
            desc=f"Disc Epoch {epoch + 1}",
        )

        for step in progress_bar:
            # Get batch
            batch_samples = discriminator_samples[step : step + args.batch_size]

            # Prepare inputs
            conflict_ids = torch.stack([s["conflict_ids"] for s in batch_samples]).to(
                accelerator.device
            )
            solution_ids = torch.stack([s["solution_ids"] for s in batch_samples]).to(
                accelerator.device
            )
            labels = torch.tensor(
                [s["is_correct"] for s in batch_samples], dtype=torch.float
            ).to(accelerator.device)
            attention_mask = torch.stack(
                [s["attention_mask"] for s in batch_samples]
            ).to(accelerator.device)

            # Forward pass
            scores = gan_model.discriminator(
                conflict_ids=conflict_ids,
                solution_ids=solution_ids,
                attention_mask=attention_mask,
            )

            # Compute loss
            loss = bce_loss(scores, labels)

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            if (step // args.batch_size + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix(
                {"disc_loss": epoch_loss / ((step // args.batch_size) + 1)}
            )

        # Evaluate discriminator performance
        logger.info(f"Discriminator Epoch {epoch + 1} completed. Running evaluation...")
        disc_metrics = evaluate_discriminator(
            gan_model, val_loader, tokenizer, accelerator
        )

        # Log metrics
        for key, value in disc_metrics.items():
            logger.info(f"Disc {key}: {value:.4f}")

        # Save best discriminator
        if disc_metrics["accuracy"] > best_score:
            best_score = disc_metrics["accuracy"]

            # Save model
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_dir, "best_discriminator")
                os.makedirs(output_dir, exist_ok=True)

                # Save discriminator
                torch.save(
                    accelerator.unwrap_model(gan_model).discriminator.state_dict(),
                    os.path.join(output_dir, "discriminator.pt"),
                )

                logger.info(f"Saved best discriminator to {output_dir}")

    return accelerator.unwrap_model(gan_model)


def train_adversarial(
    gan_model, train_loader, val_loader, tokenizer, args, accelerator
):
    """Train the full GAN adversarially"""
    # Set up optimizers
    optimizer_G = optim.AdamW(
        filter(lambda p: p.requires_grad, gan_model.generator.parameters()),
        lr=args.learning_rate,
    )

    optimizer_D = optim.AdamW(
        filter(lambda p: p.requires_grad, gan_model.discriminator.parameters()),
        lr=args.learning_rate_discriminator,
    )

    # Calculate total training steps
    _ = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps

    # Prepare for distributed training
    gan_model, optimizer_G, optimizer_D, train_loader, val_loader = accelerator.prepare(
        gan_model, optimizer_G, optimizer_D, train_loader, val_loader
    )

    # Define loss functions
    gen_criterion = nn.BCELoss()
    disc_criterion = nn.BCELoss()

    logger.info("***** Starting adversarial training *****")
    global_step = 0
    best_gan_score = 0.0

    for epoch in range(args.num_epochs):
        gan_model.train()
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"GAN Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            _ = batch["input_ids"].size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            gan_model.train_discriminator_only()

            # Generate fake solutions
            with torch.no_grad():
                fake_ids, _ = gan_model.generator.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=512,
                    num_beams=args.num_beams,
                )

            # Discriminator scores for real solutions
            real_scores = gan_model.discriminator(
                conflict_ids=batch["input_ids"],
                solution_ids=batch["labels"],
                attention_mask=batch["attention_mask"],
            )

            # Discriminator scores for fake solutions
            fake_scores = gan_model.discriminator(
                conflict_ids=batch["input_ids"],
                solution_ids=fake_ids,
                attention_mask=batch["attention_mask"],
            )

            # Create labels for discriminator
            real_labels = torch.ones_like(real_scores)
            fake_labels = torch.zeros_like(fake_scores)

            # Add noise to labels for smoother training
            if args.label_smoothing > 0:
                real_labels = real_labels - args.label_smoothing
                fake_labels = fake_labels + args.label_smoothing

            # Compute discriminator loss
            d_loss_real = disc_criterion(real_scores, real_labels)
            d_loss_fake = disc_criterion(fake_scores, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            # Scale loss for gradient accumulation
            d_loss = d_loss / args.gradient_accumulation_steps

            # Update discriminator
            accelerator.backward(d_loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer_D.step()
                optimizer_D.zero_grad()

            disc_epoch_loss += d_loss.item() * args.gradient_accumulation_steps

            # ---------------------
            # Train Generator
            # ---------------------
            if step % args.generator_steps == 0:
                gan_model.train_generator_only()

                # Forward pass through generator
                gen_outputs = gan_model.generator(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Generate fake solutions
                with torch.no_grad():
                    fake_ids, raw_confidence = gan_model.generator.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=512,
                        num_beams=args.num_beams,
                    )

                # Get discriminator scores for fake solutions
                fake_scores = gan_model.discriminator(
                    conflict_ids=batch["input_ids"],
                    solution_ids=fake_ids,
                    attention_mask=batch["attention_mask"],
                )

                # Standard generation loss
                gen_loss = gen_outputs.loss

                # Adversarial loss (fool the discriminator)
                adv_loss = gen_criterion(fake_scores, torch.ones_like(fake_scores))

                # Confidence alignment loss (confidence should match discriminator score)
                conf_loss = nn.MSELoss()(raw_confidence, fake_scores)

                # Combined loss
                g_loss = (
                    gen_loss + args.adv_weight * adv_loss + args.conf_weight * conf_loss
                )

                # Scale loss for gradient accumulation
                g_loss = g_loss / args.gradient_accumulation_steps

                # Update generator
                accelerator.backward(g_loss)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer_G.step()
                    optimizer_G.zero_grad()

                gen_epoch_loss += g_loss.item() * args.gradient_accumulation_steps

            global_step += 1
            progress_bar.set_postfix(
                {
                    "g_loss": gen_epoch_loss / (step + 1),
                    "d_loss": disc_epoch_loss / (step + 1),
                }
            )

        # Evaluate after each epoch
        logger.info(f"GAN Epoch {epoch + 1} completed. Running evaluation...")
        metrics = evaluate_gan_model(gan_model, val_loader, tokenizer, accelerator)

        # Log metrics
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        # Save best model
        if metrics["accuracy"] > best_gan_score:
            best_gan_score = metrics["accuracy"]

            # Save model
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_dir, "best_gan_model")
                os.makedirs(output_dir, exist_ok=True)

                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(gan_model)

                # Save generator
                unwrapped_model.generator.generator.save_pretrained(
                    os.path.join(output_dir, "generator")
                )

                # Save confidence head
                torch.save(
                    unwrapped_model.generator.confidence_head.state_dict(),
                    os.path.join(output_dir, "generator/confidence_head.pt"),
                )

                # Save discriminator
                torch.save(
                    unwrapped_model.discriminator.state_dict(),
                    os.path.join(output_dir, "discriminator.pt"),
                )

                logger.info(f"Saved best GAN model to {output_dir}")

    return accelerator.unwrap_model(gan_model)


def evaluate_discriminator(gan_model, val_loader, tokenizer, accelerator):
    """Evaluate discriminator performance"""
    gan_model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        # First, generate samples for evaluation
        samples = generate_samples_for_discriminator(
            gan_model,
            val_loader,
            tokenizer,
            accelerator,
            num_samples=min(1000, len(val_loader.dataset)),
        )

        # Evaluate discriminator on samples
        for i in range(0, len(samples), accelerator.state.num_processes * 4):
            batch_samples = samples[i : i + accelerator.state.num_processes * 4]

            if not batch_samples:
                continue

            # Prepare inputs
            conflict_ids = torch.stack([s["conflict_ids"] for s in batch_samples]).to(
                accelerator.device
            )
            solution_ids = torch.stack([s["solution_ids"] for s in batch_samples]).to(
                accelerator.device
            )
            labels = torch.tensor(
                [s["is_correct"] for s in batch_samples], dtype=torch.float
            ).to(accelerator.device)
            attention_mask = torch.stack(
                [s["attention_mask"] for s in batch_samples]
            ).to(accelerator.device)

            # Get discriminator scores
            scores = gan_model.discriminator(
                conflict_ids=conflict_ids,
                solution_ids=solution_ids,
                attention_mask=attention_mask,
            )

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    predictions = [score >= 0.5 for score in all_scores]
    accuracy = sum(
        [
            prediction == (label >= 0.5)
            for prediction, label in zip(predictions, all_labels)
        ]
    ) / len(predictions)

    # Calculate precision, recall, and F1 for positive class
    true_positives = sum(
        [
            prediction and label >= 0.5
            for prediction, label in zip(predictions, all_labels)
        ]
    )
    false_positives = sum(
        [
            prediction and label < 0.5
            for prediction, label in zip(predictions, all_labels)
        ]
    )
    false_negatives = sum(
        [
            not prediction and label >= 0.5
            for prediction, label in zip(predictions, all_labels)
        ]
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train merge-hesitate GAN model")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Salesforce/codet5-small",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--pretrained_generator_path",
        type=str,
        default=None,
        help="Path to pretrained generator (if available)",
    )

    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--use_augmentation", action="store_true", help="Use data augmentation"
    )

    # Training phase selection
    parser.add_argument(
        "--phase",
        type=str,
        default="generator_pretraining",
        choices=[
            "generator_pretraining",
            "discriminator_training",
            "adversarial_training",
        ],
        help="Training phase",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--learning_rate_discriminator",
        type=float,
        default=1e-5,
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # GAN specific arguments
    parser.add_argument(
        "--generator_steps", type=int, default=1, help="Steps between generator updates"
    )
    parser.add_argument(
        "--adv_weight", type=float, default=0.5, help="Weight for adversarial loss"
    )
    parser.add_argument(
        "--conf_weight",
        type=float,
        default=0.3,
        help="Weight for confidence alignment loss",
    )
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.5,
        help="Weight for confidence loss",
    )

    # Generation arguments
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Number of beams for beam search"
    )

    # Confidence arguments
    parser.add_argument(
        "--initial_threshold",
        type=float,
        default=0.8,
        help="Initial confidence threshold",
    )

    args = parser.parse_args()

    # Validate and create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train_gan(args)


if __name__ == "__main__":
    main()
