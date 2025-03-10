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

from .model import HesitateT5
from .data import create_dataloaders
from .features import FeatureExtractor
from .confidence import ConfidenceCalibrator
from .evaluate import evaluate_model

logger = logging.getLogger(__name__)

def train(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
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
        use_augmentation=args.use_augmentation
    )
    
    # Initialize model
    model = HesitateT5(model_name_or_path=args.model_name_or_path)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Define loss functions
    conf_criterion = nn.BCELoss()
    
    # Training loop
    logger.info("***** Starting training *****")
    global_step = 0
    best_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True,
                return_dict=True
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
        logger.info(f"Epoch {epoch+1} completed. Running evaluation...")
        metrics = evaluate_model(model, val_loader, tokenizer, accelerator)
        
        # Log metrics
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Save best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            
            # Save model
            if accelerator.is_main_process:
                output_dir = os.path.join(args.output_dir, f"best_model")
                os.makedirs(output_dir, exist_ok=True)
                
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.generator.save_pretrained(output_dir)
                
                # Save confidence head separately
                torch.save(unwrapped_model.confidence_head.state_dict(), 
                           os.path.join(output_dir, "confidence_head.pt"))
                
                logger.info(f"Saved best model to {output_dir}")
        
        # Train confidence calibrator after generation model is well-trained
        if epoch >= args.num_epochs // 2:
            logger.info("Training confidence calibrator...")
            train_calibrator(model, val_loader, accelerator, args)
    
    # Final evaluation on test set
    logger.info("***** Final evaluation on test set *****")
    test_loader = accelerator.prepare(test_loader)
    metrics = evaluate_model(model, test_loader, tokenizer, accelerator)
    
    # Log metrics
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("Training completed!")

def train_calibrator(model, val_loader, accelerator, args):
    """Train the confidence calibrator using validation data."""
    model.eval()
    raw_confidences = []
    correctness = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting calibration data"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get model predictions
            generated, confidence = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_seq_length,
                num_beams=args.num_beams
            )
            
            # Check if predictions match targets
            correct = []
            for i in range(len(generated)):
                gen_text = accelerator.unwrap_model(model).tokenizer.decode(
                    generated[i], skip_special_tokens=True
                )
                target_text = accelerator.unwrap_model(model).tokenizer.decode(
                    batch["labels"][i], skip_special_tokens=True
                )
                correct.append(gen_text == target_text)
            
            # Collect data for calibration
            raw_confidences.extend(confidence.cpu().tolist())
            correctness.extend(correct)
    
    # Train calibrator
    calibrator = ConfidenceCalibrator(strategy=args.calibration_strategy)
    calibrator.fit(raw_confidences, correctness)
    
    # Set new threshold in model
    accelerator.unwrap_model(model).set_confidence_threshold(calibrator.get_threshold())
    
    # Save calibrator
    if accelerator.is_main_process:
        output_dir = os.path.join(args.output_dir, "calibrator")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(calibrator, os.path.join(output_dir, "calibrator.pt"))

def main():
    parser = argparse.ArgumentParser(description="Train merge-hesitate model")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Salesforce/codet5-small", 
                      help="Path to pretrained model")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory")
    parser.add_argument("--use_augmentation", action="store_true",
                      help="Use data augmentation")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    # Generation arguments
    parser.add_argument("--num_beams", type=int, default=4,
                      help="Number of beams for beam search")
    
    # Calibration arguments
    parser.add_argument("--calibration_strategy", type=str, default="temperature",
                      choices=["temperature", "isotonic"],
                      help="Calibration strategy (temperature or isotonic)")
    parser.add_argument("--confidence_weight", type=float, default=0.5,
                      help="Weight for confidence loss")
    parser.add_argument("--initial_threshold", type=float, default=0.8,
                      help="Initial confidence threshold")
    
    args = parser.parse_args()
    
    # Validate and create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()