import torch
from difflib import SequenceMatcher
from typing import Dict
import re

class FeatureExtractor:
    """Extract features from merge conflict information to enhance model accuracy."""
    
    def __init__(self, max_seq_length=512):
        self.max_seq_length = max_seq_length
    
    def extract_features(self, conflict_text: str) -> Dict[str, torch.Tensor]:
        """
        Extract features from a merge conflict.
        
        Args:
            conflict_text: Raw text of the merge conflict including context
        
        Returns:
            Dictionary of features
        """
        # Split into parts
        parts = self._split_conflict(conflict_text)
        if not parts:
            return {"similarity": torch.tensor([0.0])}
        
        # Extract features
        features = {}
        
        # Text similarity between branches
        features["similarity"] = torch.tensor([self._compute_similarity(parts["ours"], parts["theirs"])])
        
        # Conflict complexity (length, tokens, lines)
        features["conflict_length"] = torch.tensor([len(parts["ours"] + parts["theirs"])])
        features["line_count"] = torch.tensor([
            parts["ours"].count('\n') + parts["theirs"].count('\n')
        ])
        
        # Context similarity (how similar is context to conflict)
        ctx_sim = self._compute_similarity(
            parts["pre_context"] + parts["post_context"], 
            parts["ours"] + parts["theirs"]
        )
        features["context_similarity"] = torch.tensor([ctx_sim])
        
        # Amount of overlapping content
        features["overlap"] = torch.tensor([self._compute_overlap(parts["ours"], parts["theirs"])])

        # Syntactic features for code files
        if self._is_code_file(conflict_text):
            features["code_features"] = self._extract_code_features(parts)
            
        return features
    
    def _split_conflict(self, conflict_text: str) -> Dict[str, str]:
        """Split conflict text into pre-context, ours, theirs, post-context."""
        conflict_pattern = r'<<<<<<<.*?\n(.*?)=======\n(.*?)>>>>>>>'
        match = re.search(conflict_pattern, conflict_text, re.DOTALL)
        
        if not match:
            return {}
            
        # Find the conflict boundaries
        start_idx = conflict_text.find('<<<<<<<')
        end_idx = conflict_text.find('>>>>>>>', start_idx) + 7  # +7 to include the marker
        
        # Extract context (3 lines before and after as in MergeGen)
        pre_conflict = conflict_text[:start_idx].strip()
        post_conflict = conflict_text[end_idx:].strip()
        
        # Limit context to 3 lines
        pre_lines = pre_conflict.split('\n')[-3:] if pre_conflict else []
        post_lines = post_conflict.split('\n')[:3] if post_conflict else []
        
        pre_context = '\n'.join(pre_lines)
        post_context = '\n'.join(post_lines)
        
        return {
            "pre_context": pre_context,
            "ours": match.group(1),
            "theirs": match.group(2),
            "post_context": post_context,
            "full_context": conflict_text
        }
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two pieces of text."""
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def _compute_overlap(self, text1: str, text2: str) -> float:
        """Compute ratio of overlapping lines between two texts."""
        lines1 = set(text1.split('\n'))
        lines2 = set(text2.split('\n'))
        
        if not lines1 or not lines2:
            return 0.0
            
        intersection = lines1.intersection(lines2)
        union = lines1.union(lines2)
        
        return len(intersection) / len(union)
    
    def _is_code_file(self, text: str) -> bool:
        """Determine if conflict is in a code file."""
        # Simple heuristic based on common code constructs
        code_patterns = [
            r'function\s+\w+\s*\(',  # Function definition
            r'def\s+\w+\s*\(',       # Python function
            r'class\s+\w+',          # Class definition
            r'import\s+[\w\.]+',     # Import statement
            r'#include',             # C/C++ include
            r'public|private|protected\s+\w+', # Access modifiers
            r'var|let|const\s+\w+',  # JavaScript variables
            r'return\s+[\w\(\{]'     # Return statement
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_code_features(self, parts: Dict[str, str]) -> torch.Tensor:
        """Extract code-specific features."""
        ours = parts["ours"]
        theirs = parts["theirs"]
        
        # Count common code constructs
        constructs = ["function", "class", "if", "for", "while", "return", "import", "var", "let", "const"]
        
        ours_counts = [ours.count(c) for c in constructs]
        theirs_counts = [theirs.count(c) for c in constructs]
        
        # Combine counts and differences
        feature_vector = ours_counts + theirs_counts
        feature_vector += [abs(o-t) for o, t in zip(ours_counts, theirs_counts)]
        
        # Indentation changes can be important
        ours_indent = self._avg_indentation(ours)
        theirs_indent = self._avg_indentation(theirs)
        feature_vector.append(abs(ours_indent - theirs_indent))
        
        return torch.tensor([feature_vector], dtype=torch.float)
    
    def _avg_indentation(self, text: str) -> float:
        """Calculate average indentation level in a code snippet."""
        lines = text.split('\n')
        if not lines:
            return 0.0
            
        indent_levels = []
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces)
                
        return sum(indent_levels) / len(indent_levels) if indent_levels else 0.0


def prepare_input_with_features(tokenizer, conflict_text, feature_extractor):
    """Prepare input IDs and features for the model."""
    # Tokenize text
    inputs = tokenizer(conflict_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Extract features
    features = feature_extractor.extract_features(conflict_text)
    
    return inputs, features 