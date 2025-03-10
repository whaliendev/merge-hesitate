import torch
import random
from tqdm import tqdm

def generate_samples_for_discriminator(gan_model, dataloader, tokenizer, accelerator, num_samples=1000):
    """
    Generate positive and negative samples for training the discriminator.
    
    Args:
        gan_model: MergeHesitateGAN model
        dataloader: Dataloader with training samples
        tokenizer: Tokenizer for decoding
        accelerator: Accelerator for distributed training
        num_samples: Maximum number of samples to generate
        
    Returns:
        List of sample dictionaries with conflict, solution, and correctness
    """
    gan_model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating discriminator samples"):
            # Skip if we have enough samples
            if len(samples) >= num_samples:
                break
                
            batch_size = batch["input_ids"].size(0)
            
            # Generate solutions
            generated, raw_confidence = gan_model.generator.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=512,
                num_beams=4
            )
            
            # For each sample in batch
            for i in range(batch_size):
                target_ids = batch["labels"][i]
                target_ids = target_ids[target_ids != -100]
                
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                
                is_correct = gen_text.strip() == target_text.strip()
                
                # Add positive sample (ground truth)
                samples.append({
                    "conflict_ids": batch["input_ids"][i].clone(),
                    "solution_ids": target_ids.clone(),
                    "attention_mask": batch["attention_mask"][i].clone(),
                    "is_correct": 1.0,  # Correct by definition
                })
                
                # Add real generated sample (could be correct or incorrect)
                samples.append({
                    "conflict_ids": batch["input_ids"][i].clone(),
                    "solution_ids": generated[i].clone(),
                    "attention_mask": batch["attention_mask"][i].clone(),
                    "is_correct": 1.0 if is_correct else 0.0,
                })
                
                # Generate synthetic incorrect samples
                if random.random() < 0.3:  # 30% chance to create a corrupted sample
                    # Create a corrupted solution by randomly modifying the target
                    corrupted = corrupt_solution(target_ids, tokenizer)
                    
                    samples.append({
                        "conflict_ids": batch["input_ids"][i].clone(),
                        "solution_ids": corrupted,
                        "attention_mask": batch["attention_mask"][i].clone(),
                        "is_correct": 0.0,  # Incorrect by construction
                    })
                
                # Generate ambiguous samples
                if random.random() < 0.2:  # 20% chance to create an ambiguous sample
                    parts = extract_conflict_parts(
                        tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                    )
                    
                    if parts and "ours" in parts and "theirs" in parts:
                        # Take either ours or theirs as solution
                        ambig_solution = parts["ours"] if random.random() < 0.5 else parts["theirs"]
                        ambig_ids = tokenizer(
                            ambig_solution, 
                            return_tensors="pt",
                            padding=False,
                            truncation=True
                        )["input_ids"][0].to(batch["input_ids"].device)
                        
                        is_ambig_correct = ambig_solution.strip() == target_text.strip()
                        
                        samples.append({
                            "conflict_ids": batch["input_ids"][i].clone(),
                            "solution_ids": ambig_ids,
                            "attention_mask": batch["attention_mask"][i].clone(),
                            "is_correct": 1.0 if is_ambig_correct else 0.5,  # Ambiguous samples get partial credit
                        })
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Limit to requested number
    return samples[:num_samples]

def corrupt_solution(solution_ids, tokenizer):
    """Create a corrupted version of a solution for negative samples."""
    solution = solution_ids.clone()
    
    # Skip if solution is too short
    if len(solution) < 5:
        return solution
    
    # Apply one of several corruption strategies
    strategy = random.choice([
        "token_swap",
        "token_deletion",
        "token_insertion",
        "token_repetition"
    ])
    
    if strategy == "token_swap":
        # Swap two random tokens
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2].item(), solution[idx1].item()
        
    elif strategy == "token_deletion":
        # Delete a random token
        idx = random.randint(0, len(solution) - 1)
        solution = torch.cat([solution[:idx], solution[idx+1:]])
        
    elif strategy == "token_insertion":
        # Insert a random token
        idx = random.randint(0, len(solution))
        random_token = random.randint(1000, 4000)  # Typical vocabulary range
        solution = torch.cat([solution[:idx], torch.tensor([random_token], device=solution.device), solution[idx:]])
        
    elif strategy == "token_repetition":
        # Repeat a random token
        idx = random.randint(0, len(solution) - 1)
        solution = torch.cat([solution[:idx+1], solution[idx:]])
    
    return solution

def extract_conflict_parts(conflict_text):
    """Extract the parts of a merge conflict."""
    import re
    
    # Pattern to match merge conflict
    pattern = r'<<<<<<<.*?\n(.*?)=======\n(.*?)>>>>>>>'
    match = re.search(pattern, conflict_text, re.DOTALL)
    
    if not match:
        return {}
        
    # Find the conflict boundaries
    start_idx = conflict_text.find('<<<<<<<')
    end_idx = conflict_text.find('>>>>>>>', start_idx) + 7  # +7 to include the marker
    
    # Extract parts
    pre_context = conflict_text[:start_idx].strip()
    ours = match.group(1).strip()
    theirs = match.group(2).strip()
    post_context = conflict_text[end_idx:].strip()
    
    return {
        "pre_context": pre_context,
        "ours": ours,
        "theirs": theirs,
        "post_context": post_context
    }

def create_input_from_parts(parts):
    """Create input text from conflict parts."""
    return f"{parts['pre_context']}\n<<<<<<<\n{parts['ours']}\n=======\n{parts['theirs']}\n>>>>>>>\n{parts['post_context']}" 