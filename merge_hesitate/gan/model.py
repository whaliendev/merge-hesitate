import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..model import HesitateT5
from .discriminator import MergeDiscriminator


class MergeHesitateGAN(nn.Module):
    """
    GAN-based architecture for merge conflict resolution with adversarial confidence estimation.

    Consists of:
    - Generator: Enhanced T5 model (HesitateT5) that generates merge solutions
    - Discriminator: Model that evaluates solution quality and provides refined confidence
    """

    def __init__(
        self,
        generator_path: str = "Salesforce/codet5-small",
        discriminator_path: str = "Salesforce/codet5-small",
        confidence_threshold: float = 0.8,
        pretrained_generator: Optional[HesitateT5] = None,
    ):
        super().__init__()

        # Initialize generator (use pretrained if provided)
        if pretrained_generator is not None:
            self.generator = pretrained_generator
        else:
            self.generator = HesitateT5(
                model_name_or_path=generator_path,
                confidence_threshold=confidence_threshold,
            )

        # Initialize discriminator
        self.discriminator = MergeDiscriminator(model_name_or_path=discriminator_path)

        # Store configuration
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """Forward pass through the GAN model"""
        # Generate solution with generator
        gen_outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )

        # During training, also compute discriminator scores
        if self.training and labels is not None:
            # Generate solutions (without gradient)
            with torch.no_grad():
                generated_ids, raw_confidence = self.generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                )

            # Get discriminator scores for real (ground truth) solutions
            real_score = self.discriminator(input_ids, labels, attention_mask)

            # Get discriminator scores for generated (fake) solutions
            fake_score = self.discriminator(input_ids, generated_ids, attention_mask)

            # Add scores to outputs
            if return_dict:
                gen_outputs.discriminator_real_score = real_score
                gen_outputs.discriminator_fake_score = fake_score
                gen_outputs.refined_confidence = fake_score

        return gen_outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Generate merge solutions with refined confidence estimation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for generation

        Returns:
            Tuple of (generated_ids, confidence_scores)
        """
        # First, generate solutions using the generator
        with torch.no_grad():
            generated_ids, raw_confidence = self.generator.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

            # For each generated solution with confidence above threshold,
            # refine the confidence using the discriminator
            refined_confidence = torch.zeros_like(raw_confidence)
            generation_mask = raw_confidence >= self.generator.confidence_threshold

            if generation_mask.any():
                # Only evaluate solutions that passed the initial confidence threshold
                disc_scores = self.discriminator(
                    input_ids[generation_mask],
                    generated_ids[generation_mask],
                    attention_mask[generation_mask]
                    if attention_mask is not None
                    else None,
                )

                # Update confidence scores
                refined_confidence[generation_mask] = disc_scores

            # Determine final mask based on refined confidence
            final_mask = refined_confidence >= self.confidence_threshold

            # For solutions that don't meet the refined threshold, zero out the generation
            if not final_mask.all() and generation_mask.any():
                # Create mask of generations that should be zeroed out
                zero_mask = generation_mask & ~final_mask
                if zero_mask.any():
                    generated_ids[zero_mask] = 0

        return generated_ids, refined_confidence

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for both generator and GAN"""
        self.confidence_threshold = threshold
        self.generator.set_confidence_threshold(threshold)

    def train_generator_only(self):
        """Set training mode for generator only, freeze discriminator"""
        self.train()
        self.discriminator.eval()
        for param in self.discriminator.parameters():
            param.requires_grad = False
        for param in self.generator.parameters():
            param.requires_grad = True

    def train_discriminator_only(self):
        """Set training mode for discriminator only, freeze generator"""
        self.train()
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def train_adversarial(self):
        """Set training mode for both generator and discriminator"""
        self.train()
        for param in self.generator.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True
