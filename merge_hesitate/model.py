import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import math


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
        hidden_size = self.embedding_dim

        # Feature projection layer - Now using FiLM
        if self.use_features:
             # Optional: Intermediate projection before generating scale/shift
             intermediate_feature_dim = hidden_size // 2 # Example intermediate size
             self.feature_intermediate = nn.Sequential(
                 nn.Linear(feature_size, intermediate_feature_dim),
                 nn.ReLU()
             )
             # Layers to predict scale and shift parameters
             self.feature_scale_predictor = nn.Linear(intermediate_feature_dim, hidden_size)
             self.feature_shift_predictor = nn.Linear(intermediate_feature_dim, hidden_size)
             # LayerNorm for the final fused output (optional but good practice)
             self.film_layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.feature_intermediate = None
            self.feature_scale_predictor = None
            self.feature_shift_predictor = None
            self.film_layer_norm = None

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
        """Helper to get encoder states (fused using FiLM) and confidence using Attention Pooling."""
        # Get base encoder outputs
        encoder_outputs = self.generator.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        final_encoder_states = encoder_hidden_states # Default output states

        # Fuse features using FiLM if enabled
        if self.use_features and features is not None and self.feature_intermediate is not None:
            # Project features
            intermediate_features = self.feature_intermediate(features) # [batch, intermediate_dim]
            # Predict scale and shift
            scale = self.feature_scale_predictor(intermediate_features) # [batch, hidden_size]
            shift = self.feature_shift_predictor(intermediate_features) # [batch, hidden_size]

            # Unsqueeze for broadcasting: [batch, 1, hidden_size]
            scale_unsqueezed = scale.unsqueeze(1)
            shift_unsqueezed = shift.unsqueeze(1)

            # Apply FiLM: state * (1 + scale) + shift
            # Using (1 + scale) is often more stable than just scale
            fused_states = encoder_hidden_states * (1 + scale_unsqueezed) + shift_unsqueezed

            # Apply LayerNorm after FiLM fusion
            if self.film_layer_norm is not None:
                final_encoder_states = self.film_layer_norm(fused_states)
            else:
                final_encoder_states = fused_states
        else:
            # If not using features, states remain unchanged
            final_encoder_states = encoder_hidden_states

        # Use the potentially fused states for both subsequent steps
        states_for_decoder_or_generate = final_encoder_states
        states_for_confidence = final_encoder_states

        # --- Attention Pooling Calculation (operates on states_for_confidence) ---
        batch_size = states_for_confidence.size(0)
        keys = self.key_layer(states_for_confidence)
        values = states_for_confidence
        query = self.attention_query.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)
        attention_scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.embedding_dim)
        attention_scores = attention_scores.squeeze(1)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        pooled_hidden = torch.sum(values * attention_weights.unsqueeze(-1), dim=1)
        # -------------------------------------

        # Calculate confidence based on the attention-pooled hidden state
        confidence = self.confidence_head(pooled_hidden).squeeze(-1)

        # Return potentially FiLM-fused states for decoder/generation and confidence
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

        # Compute generation loss
        log_probs = F.log_softmax(logits, dim=-1)

        shifted_labels = labels.clone()
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
            # predictions[b, t] is the model's prediction for original label token t+1
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
        encoder_outputs_for_generate = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=encoder_states_for_generate,
        )

        # Run beam search generation
        with torch.no_grad():
            outputs = self.generator.generate(
                encoder_outputs=encoder_outputs_for_generate,
                decoder_input_ids=(
                    torch.ones(len(batch[0]), 1) * self.tokenizer.bos_token_id
                )
                .long()
                .cuda(),
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams, # Generate all beams initially
                max_new_tokens=max_resolve_len * 2, # Allow longer generation before potential truncate
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
            self.feature_intermediate = None
            self.feature_scale_predictor = None
            self.feature_shift_predictor = None
            self.film_layer_norm = None
        # If enabling features later, projector needs re-initialization (not handled here)
