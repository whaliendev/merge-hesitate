import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration


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
        self.embedding_dim = self.generator.config.d_model

        # Add feature projection layer - projects feature vector to hidden dimension
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_size, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
        )

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

    def forward(self, **batch):
        """
        Forward pass with unified batch interface for simplicity.
        """
        # Extract all needed tensors from batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
        features = batch.get("features", None) if self.use_features else None
        stage = batch.get("stage", "train")

        # Get encoder outputs
        encoder_outputs = self.generator.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        # size: [batch_size, seq_len, hidden_dim]
        encoder_hidden_states = encoder_outputs["last_hidden_state"]

        # If features are provided and use_features is True, fuse them with encoder outputs
        if self.use_features and features is not None:
            # Project features to hidden dimension
            projected_features = self.feature_projector(
                features
            )  # [batch_size, hidden_dim]

            # Add as a global condition to encoder outputs
            # Method 1: Add to all positions (broadcast)
            projected_features = projected_features.unsqueeze(
                1
            )  # [batch_size, 1, hidden_dim]
            encoder_hidden_states = encoder_hidden_states + projected_features

        # Handle decoder inputs
        if labels is not None:
            # Create decoder inputs from labels for teacher forcing
            decoder_input_ids = self.generator._shift_right(labels)
        else:
            decoder_input_ids = None

        # Get decoder outputs
        decoder_outputs = self.generator.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = decoder_outputs["last_hidden_state"]

        # Apply normalization as in MergeT5
        logits = decoder_hidden_states * (self.embedding_dim**-0.5)
        # size: [batch_size, seq_len, vocab_size]
        logits = self.generator.lm_head(logits)

        # For training, compute loss
        if labels is not None:
            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Prepare shifted labels
            shifted_labels = labels.clone()
            shifted_labels = torch.cat(
                [
                    shifted_labels,
                    torch.ones(
                        (shifted_labels.size(0), 1),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    * self.tokenizer.pad_token_id,
                ],
                dim=-1,
            )
            shifted_labels = shifted_labels[:, 1:]

            # Create mask for valid positions
            mask = shifted_labels != self.tokenizer.pad_token_id

            # Compute loss
            nll_loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                shifted_labels.contiguous().view(-1),
                reduction="none",
            )
            # Correctly mask the loss using boolean inversion
            nll_loss = nll_loss.masked_fill(~mask.view(-1), 0)

            # Estimate confidence based on encoder outputs, accounting for padding
            # Use attention mask for weighted pooling
            # Add epsilon to avoid division by zero if attention_mask sum is 0
            masked_sum = (encoder_hidden_states * attention_mask.unsqueeze(-1)).sum(
                dim=1
            )
            attention_sum = attention_mask.sum(dim=1, keepdim=True)
            pooled_hidden = masked_sum / (
                attention_sum + 1e-9
            )  # Add epsilon for stability

            # batch_size, hidden_dim -> batch_size, 1 -> batch_size
            confidence = self.confidence_head(pooled_hidden).squeeze(-1)

            # Create output object
            output = {
                # size: [batch_size]
                "loss": nll_loss.sum(),
                # size: [batch_size]
                "confidence": confidence,
                # size: [batch_size, seq_len, vocab_size]
                "logits": logits,
                # size: [batch_size]
                "valid_tokens": mask.sum(),
            }

            if stage == "train":
                return output
            else:
                # For evaluation, return predicted tokens too, batch_size, seq_len
                predictions = torch.argmax(logits, dim=-1)
                # size: [batch_size, seq_len]
                output["predictions"] = predictions
                return output
        else:
            # it should never happen
            raise ValueError("labels should not be None in training stage")

    def generate(self, **batch):
        """
        Override generate method to include confidence estimation and thresholding.
        Unified batch interface for simplicity.
        """
        max_resolve_len = batch.pop("max_resolve_len", 256)

        # Get inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        features = batch.get("features", None) if self.use_features else None

        # 检查输入参数是否有效
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")

        if input_ids.dim() == 1:
            # 处理单个输入示例的情况，添加批次维度
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            if features is not None and self.use_features:
                features = features.unsqueeze(0)

        # Get encoder outputs
        encoder_outputs = self.generator.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Apply feature fusion if features provided and use_features is enabled
        if self.use_features and features is not None:
            # features: [batch_size, feature_size]
            projected_features = self.feature_projector(features)
            # projected_features: [batch_size, 1, hidden_dim]
            projected_features = projected_features.unsqueeze(1)
            # broadcast on seq_len dimension
            encoder_hidden_states = encoder_hidden_states + projected_features

            # Update encoder_outputs with modified hidden states
            encoder_outputs["last_hidden_state"] = encoder_hidden_states

        # Get confidence, [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        pooled_hidden = encoder_hidden_states.mean(dim=1)
        # [batch_size, hidden_dim] -> [batch_size, 1] -> [batch_size]
        confidence = self.confidence_head(pooled_hidden).squeeze(-1)

        # 确保没有修改原始batch
        generation_kwargs = batch.copy()
        # 移除已处理的参数
        generation_kwargs.pop("input_ids", None)
        generation_kwargs.pop("attention_mask", None)
        generation_kwargs.pop("features", None)

        # Use T5's generate with our encoder outputs
        with torch.no_grad():
            # outputs: [batch_size, seq_len]
            outputs = self.generator.generate(
                encoder_outputs=encoder_outputs,
                **generation_kwargs,
                max_new_tokens=max_resolve_len * 2,
            )

        # 设置最终输出的大小为max_resolve_len
        final_outputs_size = max_resolve_len

        # 创建最终输出张量，用于储存处理后的生成结果
        pad_token_id = (
            self.generator.config.pad_token_id
            if hasattr(self.generator.config, "pad_token_id")
            else 0
        )

        # 处理生成结果：截断或填充到统一长度
        if outputs.size(1) > final_outputs_size:
            # 截断过长的序列
            final_outputs = outputs[:, :final_outputs_size]
        else:
            # 填充过短的序列
            final_outputs = (
                torch.ones(
                    (outputs.size(0), final_outputs_size),
                    dtype=torch.long,
                    device=outputs.device,
                )
                * pad_token_id
            )
            final_outputs[:, : outputs.size(1)] = outputs

        return final_outputs, confidence

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for generation."""
        self.confidence_threshold = threshold

    def set_use_features(self, use_features):
        """Enable or disable feature usage."""
        self.use_features = use_features
