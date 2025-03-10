import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from typing import Optional, Dict


class HesitateT5(nn.Module):
    """
    Enhanced T5 model for merge conflict resolution with confidence estimation.
    Based on CodeT5-small, this model adds a confidence estimation head.
    """

    def __init__(
        self, model_name_or_path="Salesforce/codet5-small", confidence_threshold=0.8
    ):
        super().__init__()
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

        # Confidence estimation head
        hidden_size = self.generator.config.d_model
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        feature_inputs: Optional[Dict[str, torch.Tensor]] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        # Generate solution with T5
        gen_outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # Extract encoder's last hidden state for confidence estimation
        if output_hidden_states:
            encoder_last_hidden_state = gen_outputs.encoder_last_hidden_state
            # Global pooling: average across sequence length
            pooled_hidden = encoder_last_hidden_state.mean(dim=1)

            # Estimate confidence
            confidence = self.confidence_head(pooled_hidden).squeeze(-1)
        else:
            confidence = torch.ones(input_ids.size(0), device=input_ids.device) * 0.5

        # Add confidence to outputs
        if return_dict:
            gen_outputs.confidence = confidence

        return gen_outputs

    def generate(self, *args, **kwargs):
        """
        Override generate method to include confidence estimation and thresholding.
        """
        # Run normal generation first
        with torch.no_grad():
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")

            # Get encoder outputs for confidence estimation
            encoder_outputs = self.generator.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

            # Get confidence
            pooled_hidden = encoder_outputs.last_hidden_state.mean(dim=1)
            confidence = self.confidence_head(pooled_hidden).squeeze(-1)

            # Only generate for inputs with confidence above threshold
            generation_mask = confidence >= self.confidence_threshold

            if generation_mask.any():
                # Generate for confident examples
                outputs = self.generator.generate(
                    input_ids=input_ids[generation_mask],
                    attention_mask=attention_mask[generation_mask]
                    if attention_mask is not None
                    else None,
                    **kwargs,
                )
            else:
                # Return empty tensor if no examples are confident enough
                outputs = torch.zeros(
                    (input_ids.size(0), 1), dtype=torch.long, device=input_ids.device
                )

        # Prepare final outputs with appropriate shape
        final_outputs = torch.zeros(
            (input_ids.size(0), outputs.size(1) if outputs.size(0) > 0 else 1),
            dtype=torch.long,
            device=input_ids.device,
        )

        # Fill in generations for confident examples
        if generation_mask.any():
            final_outputs[generation_mask] = outputs

        return final_outputs, confidence

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for generation."""
        self.confidence_threshold = threshold
