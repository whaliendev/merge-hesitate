import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class HesitateT5(nn.Module):
    """
    Enhanced T5 model for merge conflict resolution with confidence estimation.
    Based on CodeT5-small, this model adds a confidence estimation head.
    """

    def __init__(
        self,
        model_name_or_path="./codet5-small",
        confidence_threshold=0.8,
        tokenizer=None,
        feature_size=12,
        use_features=True,
    ):
        super().__init__()
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer  # Store tokenizer if provided
        self.use_features = use_features  # Flag to control feature usage

        # If tokenizer was provided and has additional tokens, resize embeddings
        if tokenizer is not None:
            self.generator.resize_token_embeddings(len(tokenizer))

        # Get embedding dimension
        self.embedding_dim = self.generator.config.hidden_size

        # Add feature projection layer - projects feature vector to hidden dimension
        # features extracted should be add both to generation and confidence estimation
        if self.use_features:
             self.feature_projector = nn.Sequential(
                nn.Linear(feature_size, self.embedding_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embedding_dim),
            )
        else:
            # If not using features, set projector to None
            self.feature_projector = None

        # Confidence estimation head
        hidden_size = self.embedding_dim
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.confidence_threshold = confidence_threshold

    def _get_fused_encoder_states_and_confidence(self, input_ids, attention_mask, features):
        """Helper to get encoder states (potentially fused) and confidence."""
        # Get base encoder outputs
        encoder_outputs = self.generator.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        states_for_decoder_or_generate = encoder_hidden_states # Default

        # If features are provided and use_features is True, fuse them
        if self.use_features and features is not None and self.feature_projector is not None:
            # Project features: [batch_size, feature_size] -> [batch_size, hidden_dim]
            projected_features = self.feature_projector(features)
            # Add as a global condition: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            projected_features_unsqueezed = projected_features.unsqueeze(1)
            # Fuse: [batch_size, seq_len, hidden_dim] + [batch_size, 1, hidden_dim] -> [batch_size, seq_len, hidden_dim]
            fused_encoder_hidden_states = encoder_hidden_states + projected_features_unsqueezed
            # Use fused states for subsequent steps
            states_for_decoder_or_generate = fused_encoder_hidden_states
            states_for_confidence = fused_encoder_hidden_states # Use fused for confidence too
        else:
            # Use original states if no features or not using them
            states_for_confidence = encoder_hidden_states

        # Calculate confidence based on the appropriate states (potentially fused)
        # Use attention mask for weighted pooling
        masked_sum = (states_for_confidence * attention_mask.unsqueeze(-1)).sum(dim=1)
        attention_sum = attention_mask.sum(dim=1, keepdim=True)
        pooled_hidden = masked_sum / (attention_sum + 1e-9) # Add epsilon for stability
        confidence = self.confidence_head(pooled_hidden).squeeze(-1) # [batch_size]

        # Return potentially fused states for decoder/generation and confidence
        return states_for_decoder_or_generate, confidence, encoder_outputs

    def forward(self, **batch):
        """
        Forward pass for training and validation stages. Always expects labels.
        """
        # Extract required inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")
        features = batch.get("features", None) if self.use_features else None
        stage = batch.get("stage", "train") # Default to train if not specified

        if labels is None:
            raise ValueError("Labels must be provided for forward pass (train/val stages).")

        # Get potentially fused encoder states and confidence score
        encoder_states_for_decoder, confidence, _ = self._get_fused_encoder_states_and_confidence(
            input_ids, attention_mask, features
        )

        # Prepare decoder inputs
        decoder_input_ids = self.generator._shift_right(labels)

        # Get decoder outputs using potentially fused encoder states
        decoder_outputs = self.generator.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_states_for_decoder, # Use potentially fused states
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = decoder_outputs["last_hidden_state"]

        # Apply normalization and LM head
        logits = decoder_hidden_states * (self.embedding_dim**-0.5)
        logits = self.generator.lm_head(logits) # [batch_size, seq_len, vocab_size]

        # Compute generation loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Prepare shifted labels for loss calculation
        shifted_labels = labels.clone()
        # Ensure pad_token_id is retrieved correctly
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else self.generator.config.pad_token_id
        shifted_labels = torch.cat(
            [
                shifted_labels,
                torch.ones(
                    (shifted_labels.size(0), 1),
                    dtype=torch.long,
                    device=input_ids.device,
                ) * pad_token_id,
            ],
            dim=-1,
        )
        shifted_labels = shifted_labels[:, 1:]

        # Create mask for valid label positions (non-padding)
        mask = shifted_labels != pad_token_id

        # Compute NLL loss, masked for padding
        nll_loss = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            shifted_labels.contiguous().view(-1),
            reduction="none",
        )
        nll_loss = nll_loss.masked_fill(~mask.view(-1), 0)

        # Prepare output dictionary
        output = {
            "loss": nll_loss.sum(),               # Sum of loss values for the batch
            "confidence": confidence,             # [batch_size]
            "logits": logits,                     # [batch_size, seq_len, vocab_size]
            "valid_tokens": mask.sum(),           # Total number of non-padded tokens in batch labels
        }

        # Add predictions for validation stage
        if stage == "val":
            predictions = torch.argmax(logits, dim=-1) # [batch_size, seq_len]
            output["predictions"] = predictions

        return output


    def generate(self, **batch):
        """
        Generate sequences for the test stage using beam search.
        Does not expect labels. Calculates confidence.
        """
        # Extract required inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        features = batch.get("features", None) if self.use_features else None

        # Generation specific parameters (remove from batch to avoid passing to T5 generate)
        max_resolve_len = batch.pop("max_resolve_len", 256)
        num_beams = batch.pop("num_beams", 3) # Default to 3 beams like old code
        # Any other kwargs intended for self.generator.generate should remain in batch

        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")

        # Handle single input example (add batch dimension)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            if features is not None and self.use_features:
                features = features.unsqueeze(0)

        # Ensure attention_mask exists if None was passed
        if attention_mask is None:
            raise ValueError("attention_mask should not be None in generation stage")


        # Get potentially fused encoder states and confidence score
        # Note: _get_fused_encoder_states_and_confidence returns the states
        # suitable for the decoder/generator, which is what we need here.
        encoder_states_for_generate, confidence, original_encoder_outputs = self._get_fused_encoder_states_and_confidence(
            input_ids, attention_mask, features
        )

        # Prepare encoder_outputs object for T5 generate
        # T5 generate expects an object with a last_hidden_state attribute
        encoder_outputs_for_generate = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=encoder_states_for_generate,
        )

        # Run beam search generation
        with torch.no_grad():
            # Pass the rest of the batch kwargs (like max_length)
            outputs = self.generator.generate(
                encoder_outputs=encoder_outputs_for_generate,
                attention_mask=attention_mask, # Pass attention mask explicitly
                decoder_input_ids = (
                    torch.ones_like(input_ids.size(0), 1) * self.tokenizer.bos_token_id
                ).long().cuda(),
                num_beams=num_beams,
                num_return_sequences=num_beams, # Generate all beams initially
                max_new_tokens=max_resolve_len * 2, # Allow longer generation before potential truncate
            )

        # Reshape output for beam search: [batch_size * num_beams, seq_len] -> [batch_size, num_beams, seq_len]
        outputs = outputs.view(input_ids.size(0), num_beams, -1)

        # Select the top beam (index 0) - aligning with old run_mergegen.py test_beam logic
        # Shape: [batch_size, seq_len]
        top_beam_output = outputs[:, 0, 1:]

        # Pad or truncate the top beam output to max_resolve_len
        final_outputs_size = max_resolve_len
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else self.generator.config.pad_token_id

        if top_beam_output.size(1) >= final_outputs_size:
            # Truncate
            final_outputs = top_beam_output[:, :final_outputs_size]
        else:
            # Pad
            padding_size = final_outputs_size - top_beam_output.size(1)
            padding = torch.ones(
                (top_beam_output.size(0), padding_size),
                dtype=torch.long,
                device=top_beam_output.device,
            ) * pad_token_id
            final_outputs = torch.cat([top_beam_output, padding], dim=1)


        # Return the processed top beam sequence and the confidence score
        return final_outputs, confidence

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for generation."""
        self.confidence_threshold = threshold

    def set_use_features(self, use_features):
        """Enable or disable feature usage."""
        self.use_features = use_features
        if not use_features:
            self.feature_projector = None # Ensure projector is None if features disabled
        # If enabling features later, projector needs re-initialization (not handled here)
