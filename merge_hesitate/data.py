import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from transformers import PreTrainedTokenizer
from .features import FeatureExtractor

class MergeConflictDataset(Dataset):
    """Dataset for merge conflict resolution."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer,
        feature_extractor: FeatureExtractor,
        max_seq_length: int = 512,
        split: str = "train",
        use_augmentation: bool = False
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_seq_length = max_seq_length
        self.use_augmentation = use_augmentation
        
        # Load data
        data_file = os.path.join(data_path, f"{split}.json")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
            
        # Optionally augment training data
        if split == "train" and use_augmentation:
            self._augment_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input text (context + conflict markers)
        input_text = self._format_input(item)
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Extract features
        features = self.feature_extractor.extract_features(input_text)
        
        # Prepare target text (solution)
        if "solution" in item:
            target = self.tokenizer(
                item["solution"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            inputs["labels"] = target["input_ids"].squeeze(0)
            
            # Add correctness flag for training
            inputs["is_correct"] = torch.tensor(item.get("is_correct", True), dtype=torch.float)
        
        # Add features to inputs
        for k, v in features.items():
            inputs[f"feature_{k}"] = v
            
        return inputs
    
    def _format_input(self, item):
        """Format input text with conflict markers and context."""
        pre_context = item.get("pre_context", "")
        post_context = item.get("post_context", "")
        ours = item.get("ours", "")
        theirs = item.get("theirs", "")
        
        formatted = (
            f"{pre_context}\n"
            f"<<<<<<< OURS\n{ours}\n"
            f"=======\n{theirs}\n"
            f">>>>>>> THEIRS\n"
            f"{post_context}"
        )
        
        return formatted.strip()
    
    def _augment_data(self):
        """Augment training data with harder examples."""
        augmented = []
        
        for item in self.data:
            # Skip items without solutions
            if "solution" not in item:
                continue
                
            # Create a hard negative example by swapping ours and theirs
            if random.random() < 0.3:  # 30% chance to create a swap
                new_item = item.copy()
                new_item["ours"] = item["theirs"]
                new_item["theirs"] = item["ours"]
                # Keep the same solution but mark as ambiguous
                new_item["is_ambiguous"] = True
                augmented.append(new_item)
            
            # Create examples where taking just one side is correct
            if random.random() < 0.2:  # 20% chance
                if item["solution"] == item["ours"]:
                    # Solution equals "ours" side - make a clearer example
                    new_item = item.copy()
                    new_item["is_correct"] = True
                    augmented.append(new_item)
                elif item["solution"] == item["theirs"]:
                    # Solution equals "theirs" side - make a clearer example
                    new_item = item.copy()
                    new_item["is_correct"] = True
                    augmented.append(new_item)
        
        # Add augmented examples to dataset
        self.data.extend(augmented)

def create_dataloaders(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    feature_extractor: FeatureExtractor,
    batch_size: int = 8,
    max_seq_length: int = 512,
    use_augmentation: bool = True
):
    """Create dataloaders for training and evaluation."""
    train_dataset = MergeConflictDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_seq_length=max_seq_length,
        split="train",
        use_augmentation=use_augmentation
    )
    
    val_dataset = MergeConflictDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_seq_length=max_seq_length,
        split="val",
        use_augmentation=False
    )
    
    test_dataset = MergeConflictDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_seq_length=max_seq_length,
        split="test",
        use_augmentation=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 