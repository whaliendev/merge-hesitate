import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional

class MergeDiscriminator(nn.Module):
    """
    Discriminator model that evaluates the quality of merge conflict solutions.
    
    The discriminator assesses whether a given solution correctly resolves
    the merge conflict by evaluating multiple quality aspects.
    """
    
    def __init__(self, model_name_or_path: str = "Salesforce/codet5-small"):
        super().__init__()
        
        # Initialize encoder from pretrained model
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.encoder.config.d_model
        
        # Multi-aspect quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 512),  # 3x for conflict, solution, and interaction
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # 4 quality dimensions
        )
        
        # Dimension weights - higher weight for overall quality
        self.dimension_weights = nn.Parameter(
            torch.tensor([1.5, 0.8, 1.0, 0.7], dtype=torch.float)
        )
    
    def forward(
        self, 
        conflict_ids: torch.LongTensor,
        solution_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Evaluate the quality of a merge solution.
        
        Args:
            conflict_ids: Token IDs of the merge conflict input
            solution_ids: Token IDs of the proposed solution
            attention_mask: Optional attention mask for the conflict
            
        Returns:
            Confidence score (0-1) for the solution quality
        """
        batch_size = conflict_ids.size(0)
        
        # Encode conflict
        conflict_outputs = self.encoder(
            input_ids=conflict_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get conflict representation (mean pooling)
        conflict_mask = attention_mask.unsqueeze(-1) if attention_mask is not None else 1.0
        conflict_rep = (conflict_outputs.last_hidden_state * conflict_mask).sum(dim=1) / conflict_mask.sum(dim=1)
        
        # Create solution attention mask (ignoring padding)
        solution_mask = (solution_ids != 0).float()
        
        # Encode solution
        solution_outputs = self.encoder(
            input_ids=solution_ids,
            attention_mask=solution_mask,
            return_dict=True
        )
        
        # Get solution representation (mean pooling)
        solution_mask = solution_mask.unsqueeze(-1)
        solution_rep = (solution_outputs.last_hidden_state * solution_mask).sum(dim=1) / solution_mask.sum(dim=1).clamp(min=1.0)
        
        # Combine representations with element-wise product for interaction features
        combined_rep = torch.cat([
            conflict_rep, 
            solution_rep, 
            conflict_rep * solution_rep  # interaction features
        ], dim=1)
        
        # Compute quality scores for multiple dimensions
        # [0]: Overall solution quality
        # [1]: Syntactic correctness
        # [2]: Semantic preservation
        # [3]: Conflict resolution completeness
        quality_scores = self.quality_head(combined_rep)
        
        # Apply sigmoid to get scores in range [0,1]
        quality_scores = torch.sigmoid(quality_scores)
        
        # Compute weighted score (using learnable weights)
        norm_weights = nn.functional.softmax(self.dimension_weights, dim=0)
        weighted_score = (quality_scores * norm_weights).sum(dim=1)
        
        return weighted_score
    
    def compute_detailed_scores(
        self, 
        conflict_ids: torch.LongTensor,
        solution_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute detailed quality scores across all dimensions.
        
        Args:
            conflict_ids: Token IDs of the merge conflict input
            solution_ids: Token IDs of the proposed solution
            attention_mask: Optional attention mask for the conflict
            
        Returns:
            Tensor of quality scores across 4 dimensions
        """
        # Encode conflict
        conflict_outputs = self.encoder(
            input_ids=conflict_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get conflict representation (mean pooling)
        conflict_mask = attention_mask.unsqueeze(-1) if attention_mask is not None else 1.0
        conflict_rep = (conflict_outputs.last_hidden_state * conflict_mask).sum(dim=1) / conflict_mask.sum(dim=1)
        
        # Create solution attention mask (ignoring padding)
        solution_mask = (solution_ids != 0).float()
        
        # Encode solution
        solution_outputs = self.encoder(
            input_ids=solution_ids,
            attention_mask=solution_mask,
            return_dict=True
        )
        
        # Get solution representation (mean pooling)
        solution_mask = solution_mask.unsqueeze(-1)
        solution_rep = (solution_outputs.last_hidden_state * solution_mask).sum(dim=1) / solution_mask.sum(dim=1).clamp(min=1.0)
        
        # Combine representations
        combined_rep = torch.cat([
            conflict_rep, 
            solution_rep, 
            conflict_rep * solution_rep
        ], dim=1)
        
        # Compute quality scores
        quality_scores = self.quality_head(combined_rep)
        
        # Apply sigmoid to get scores in range [0,1]
        return torch.sigmoid(quality_scores) 