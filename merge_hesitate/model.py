import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import math # Needed for sqrt in attention


class HesitateT5(nn.Module):
    """
    Enhanced T5 model for merge conflict resolution with confidence estimation.
    Uses Attention Pooling for confidence head input.
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
        hidden_size = self.embedding_dim # Alias for clarity

        # Feature projection layer
        if self.use_features:
             self.feature_projector = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.feature_projector = None

        # --- Attention Pooling Components ---
        # Learnable query vector for attention pooling
        self.attention_query = nn.Parameter(torch.randn(hidden_size))
        # Linear layer to project encoder states into keys for attention
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        # Optional: Add value layer if needed, but often values are the original states
        # self.value_layer = nn.Linear(hidden_size, hidden_size)
        # ----------------------------------

        # Confidence estimation head (takes pooled output)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.confidence_threshold = confidence_threshold

    def _get_fused_encoder_states_and_confidence(self, input_ids, attention_mask, features):
        """Helper to get encoder states (potentially fused) and confidence using Attention Pooling."""
        # Get base encoder outputs
        encoder_outputs = self.generator.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        states_for_decoder_or_generate = encoder_hidden_states # Default

        # Fuse features if enabled
        if self.use_features and features is not None and self.feature_projector is not None:
            projected_features = self.feature_projector(features)
            projected_features_unsqueezed = projected_features.unsqueeze(1)
            fused_encoder_hidden_states = encoder_hidden_states + projected_features_unsqueezed
            states_for_decoder_or_generate = fused_encoder_hidden_states
            states_for_confidence = fused_encoder_hidden_states
        else:
            states_for_confidence = encoder_hidden_states

        # --- Attention Pooling Calculation ---
        batch_size = states_for_confidence.size(0)

        # Project states into keys
        keys = self.key_layer(states_for_confidence) # [batch_size, seq_len, hidden_size]
        # Values are often the original states (or could be projected)
        values = states_for_confidence # [batch_size, seq_len, hidden_size]

        # Prepare query: [hidden_size] -> [batch_size, 1, hidden_size]
        query = self.attention_query.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)

        # Calculate attention scores: (query * keys^T) / sqrt(d_k)
        # Matmul: [b, 1, h] @ [b, h, s] -> [b, 1, s]
        attention_scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.embedding_dim)
        attention_scores = attention_scores.squeeze(1) # [batch_size, seq_len]

        # Apply attention mask (set padding tokens to -inf)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Calculate attention weights
        attention_weights = F.softmax(attention_scores, dim=1) # [batch_size, seq_len]

        # Compute weighted sum of values
        # Sum: sum(weights[b, s] * values[b, s, h]) over s -> [b, h]
        pooled_hidden = torch.sum(values * attention_weights.unsqueeze(-1), dim=1) # [batch_size, hidden_size]
        # -------------------------------------

        # Calculate confidence based on the attention-pooled hidden state
        confidence = self.confidence_head(pooled_hidden).squeeze(-1) # [batch_size]

        # Return potentially fused states for decoder/generation and confidence
        return states_for_decoder_or_generate, confidence, encoder_outputs

    def forward(self, **batch):
        """
        Forward pass for training and validation stages. Always expects labels.
        Mimics run_mergegen.py by feeding original labels (WITH BOS) to decoder
        and manually shifting labels BEFORE loss calculation.
        """
        # Extract required inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels") # Expects labels WITH initial BOS from data loader
        features = batch.get("features", None) if self.use_features else None
        stage = batch.get("stage", "train") # Default to train if not specified

        if labels is None:
            raise ValueError("Labels must be provided for forward pass (train/val stages).")

        # Get potentially fused encoder states and confidence score (now uses Attention Pooling)
        encoder_states_for_decoder, confidence, _ = self._get_fused_encoder_states_and_confidence(
            input_ids, attention_mask, features
        )

        # Prepare decoder inputs - Feed ORIGINAL labels (WITH BOS) directly
        decoder_input_ids = labels

        # Get decoder outputs using potentially fused encoder states
        decoder_outputs = self.generator.decoder(
            input_ids=decoder_input_ids, # Feed original labels (WITH BOS) here
            encoder_hidden_states=encoder_states_for_decoder, # Use potentially fused states
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = decoder_outputs["last_hidden_state"]

        # Apply normalization and LM head
        logits = decoder_hidden_states * (self.embedding_dim**-0.5)
        logits = self.generator.lm_head(logits) # [batch_size, seq_len, vocab_size]
        # logits[b, t, :] is prediction for token t+1 based on input labels[0...t]

        # Compute generation loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Prepare shifted labels for loss calculation (Mimic run_mergegen.py)
        shifted_labels = labels.clone()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else self.generator.config.pad_token_id
        shifted_labels = torch.cat( # Append PAD
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
        shifted_labels = shifted_labels[:, 1:] # shifted_labels[b, t] is original label[b, t+1]

        # Create mask for valid label positions (non-padding) based on SHIFTED labels
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
            "loss": nll_loss.sum(),
            "confidence": confidence,
            "logits": logits,
            "valid_tokens": mask.sum(),
        }

        # Add predictions for validation stage
        if stage == "val":
            predictions = torch.argmax(logits, dim=-1)
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


        # Get potentially fused encoder states and confidence score (now uses Attention Pooling)
        encoder_states_for_generate, confidence, original_encoder_outputs = self._get_fused_encoder_states_and_confidence(
            input_ids, attention_mask, features
        )

        # Prepare encoder_outputs object for T5 generate
        encoder_outputs_for_generate = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=encoder_states_for_generate,
        )

        # Run beam search generation
        with torch.no_grad():
            outputs = self.generator.generate(
                encoder_outputs=encoder_outputs_for_generate,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams, # Generate all beams initially
                max_new_tokens=max_resolve_len * 2, # Allow longer generation before potential truncate
                **batch # Pass remaining kwargs like max_length etc.
            )

        # Reshape output for beam search: [batch_size * num_beams, seq_len] -> [batch_size, num_beams, seq_len]
        outputs = outputs.view(input_ids.size(0), num_beams, -1)

        # Select the top beam (index 0)
        top_beam_output = outputs[:, 0, 1:] # Remove potential leading BOS/PAD from generate output

        # Pad or truncate the top beam output to max_resolve_len
        final_outputs_size = max_resolve_len
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else self.generator.config.pad_token_id

        if top_beam_output.size(1) >= final_outputs_size:
            final_outputs = top_beam_output[:, :final_outputs_size]
        else:
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
