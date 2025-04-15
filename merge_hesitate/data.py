import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
from transformers import PreTrainedTokenizer
from accelerate import Accelerator

class MergeConflictDataset(Dataset):
    """Dataset for merge conflict resolution using preprocessed data."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        max_resolve_length: int = 256,
        split: str = "train",
        feature_size: int = 12,
        use_features: bool = True,
        accelerator: Accelerator = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_conflict_length = (
            max_seq_length - feature_size if use_features else max_seq_length
        )  # Reserve space for features only if needed
        self.max_resolve_length = max_resolve_length  # As defined in preprocessing
        self.feature_size = feature_size
        self.split = split
        self.use_features = use_features

        # Load preprocessed data
        preprocessed_path = f"PREPROCESSED/processed_{split}.pkl"
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(
                f"Preprocessed data not found at {preprocessed_path}. Run preprocessing first."
            )

        with open(preprocessed_path, "rb") as f:
            self.data = pickle.load(f)

        # data[0] is input_ids (conflicts), data[1] is output_ids (resolutions)
        self.inputs = self.data[0]
        self.outputs = self.data[1]

        # Pre-compute features only if we're using them
        self.features = self._precompute_features() if use_features else None

        accelerator.print(f"Loaded {len(self.inputs)} samples from {preprocessed_path}")
        if use_features:
            accelerator.print(f"Feature extraction enabled with {feature_size} features")
        else:
            accelerator.print("Feature extraction disabled")

    def _precompute_features(self):
        """Precompute features for all samples in the dataset."""
        features = []
        extractor = FeatureExtractor(self.max_seq_length)

        for i in range(len(self.inputs)):
            # This is just a placeholder - in a real implementation,
            # you would extract meaningful features from the actual data
            sample_features = extractor.extract_features(
                {"input_ids": self.inputs[i], "output_ids": self.outputs[i], "idx": i}
            )
            features.append(
                torch.tensor(list(sample_features.values()), dtype=torch.float)
            )

        return features

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Get preprocessed input and output sequences
        input_ids = self.inputs[idx]
        output_ids = self.outputs[idx]

        # Create attention mask (1 for non-pad tokens)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(np.int64)

        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
        output_tensor = torch.tensor(output_ids, dtype=torch.long)

        # Create sample with all required fields
        sample = {
            "input_ids": input_tensor,
            "attention_mask": attention_tensor,
            "labels": output_tensor,
        }

        # Add features if enabled
        if self.use_features:
            feature_tensor = self.features[idx]
            sample["features"] = feature_tensor
        else:
            # Add dummy features tensor with zeros if needed by the model
            # This ensures compatibility even when features are disabled
            sample["features"] = torch.zeros(self.feature_size, dtype=torch.float)

        return sample


class FeatureExtractor:
    """Extract features from merge conflict tokens."""

    def __init__(self, max_seq_length=512):
        self.max_seq_length = max_seq_length

    def extract_features(self, sample_data):
        """
        Extract features from tokens.

        Args:
            sample_data: Dictionary containing input_ids, output_ids, and other info

        Returns:
            Dictionary of features
        """
        # Here you could compute actual similarity metrics between
        # parts of the merge conflict (using the <lbra>, <rbra>, etc. tokens)
        # As a placeholder, we'll return some random features

        # Using the index to make different samples have different features
        idx = sample_data.get("idx", 0)
        seed = (
            idx * 10
        ) % 100  # A simple way to generate different features per sample

        # In a real implementation, you would:
        # 1. Extract OURS section from input_ids (between <lbra> and separator)
        # 2. Extract THEIRS section from input_ids (between separators and <rbra>)
        # 3. Extract BASE section from input_ids (between separators)
        # 4. Compute similarity metrics between these sections

        # Generate slightly different features based on sample index
        np.random.seed(seed)

        features = {
            "conflict_similarity": np.clip(0.5 + np.random.normal(0, 0.1), 0, 1),
            "base_ours_similarity": np.clip(0.7 + np.random.normal(0, 0.1), 0, 1),
            "base_theirs_similarity": np.clip(0.6 + np.random.normal(0, 0.1), 0, 1),
            "conflict_length_ratio": np.clip(0.4 + np.random.normal(0, 0.1), 0, 1),
            "token_overlap": np.clip(0.3 + np.random.normal(0, 0.1), 0, 1),
            # Add more features as needed, up to feature_size
        }

        return features


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 36,
    max_seq_length: int = 512,
    max_resolve_length: int = 256,
    feature_size: int = 12,
    use_features: bool = True,
    accelerator: Accelerator = None,
):
    """Create dataloaders for training and evaluation."""
    train_dataset = MergeConflictDataset(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_resolve_length=max_resolve_length,
        split="train",
        feature_size=feature_size,
        use_features=use_features,
        accelerator=accelerator,
    )

    val_dataset = MergeConflictDataset(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_resolve_length=max_resolve_length,
        split="val",
        feature_size=feature_size,
        use_features=use_features,
        accelerator=accelerator,
    )

    test_dataset = MergeConflictDataset(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_resolve_length=max_resolve_length,
        split="test",
        feature_size=feature_size,
        use_features=use_features,
        accelerator=accelerator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
