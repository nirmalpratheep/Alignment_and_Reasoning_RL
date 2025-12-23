"""Training utilities with loss tracking and W&B logging"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from typing import List, Dict, Tuple


LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_sft_batch(batch: List[Dict], tokenizer, max_length: int = 1024) -> Dict:
    """Prepare a batch of SFT data for training
    
    Args:
        batch: List of training examples with 'prompt' and 'response' keys
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
    
    Returns:
        Dict: Contains input_ids, attention_mask, labels
    """
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    
    # Combine prompt and response
    texts = [p + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)
    
    # Labels: same as input_ids (for causal language modeling)
    labels = input_ids.clone()
    
    # Mask out prompt tokens so we only compute loss on response
    prompt_encodings = tokenizer(
        prompts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    prompt_lengths = (prompt_encodings['attention_mask'].sum(dim=1)).tolist()
    
    # Set labels to -100 for prompt tokens (ignored by loss function)
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def train_sft_model(model, tokenizer, train_data: List[Dict], val_data: List[Dict],
                    batch_size: int = 4, num_epochs: int = 1, 
                    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
                    learning_rate: float = LEARNING_RATE, max_batches: int = 5) -> None:
    """Train model using SFT on the prepared dataset with W&B logging
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for encoding
        train_data: Training data with 'prompt' and 'response' keys
        val_data: Validation data
        batch_size: Batch size per step
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate for optimizer
        max_batches: Maximum number of batches to process (default: 5)
    """
    print("\n" + "="*80)
    print("STARTING SFT MODEL TRAINING")
    print("="*80)
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Enable memory optimizations
    print("\nEnabling memory optimizations...")
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    model.enable_input_require_grads()  # Required for gradient checkpointing
    print("✓ Gradient checkpointing enabled (reduces memory at cost of compute)")
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Setup learning rate scheduler
    num_training_steps = (len(train_data) // batch_size) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    # Setup mixed precision (BF16)
    from torch.cuda.amp import autocast
    print("✓ Using BF16 mixed precision for training")
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        epoch_train_loss = 0.0
        batch_count = 0
        accumulated_loss = 0.0
        
        # Create batches - limit to max_batches
        num_batches_to_process = min(max_batches, len(train_data) // batch_size)
        progress_bar = tqdm(range(num_batches_to_process), desc="Training batches")
        
        for batch_num in progress_bar:
            batch_idx = batch_num * batch_size
            batch = train_data[batch_idx:batch_idx + batch_size]
            
            try:
                # Prepare batch
                batch_data = prepare_sft_batch(batch, tokenizer)
                input_ids = batch_data['input_ids']
                attention_mask = batch_data['attention_mask']
                labels = batch_data['labels']
                
                # Forward pass with BF16 mixed precision
                with autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True
                    )
                
                loss = outputs.loss
                
                # Normalize loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                accumulated_loss += loss.item()
                
                # Optimizer step after accumulation
                if (batch_num + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Accumulate metrics
                    batch_loss = accumulated_loss
                    epoch_train_loss += batch_loss
                    batch_count += 1
                    global_step += 1
                    accumulated_loss = 0.0
                    
                    # Log batch metrics to W&B
                    wandb.log({
                        'train/batch_loss': batch_loss,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/global_step': global_step,
                        'train/epoch': epoch + 1,
                    })
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue
        
        # Final optimizer step if needed
        if accumulated_loss > 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Calculate epoch training loss
        avg_epoch_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0.0
        
        # Validation phase - limit to max_batches
        print(f"\nValidating on validation examples (max {max_batches} batches)...")
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            num_val_batches = min(max_batches, len(val_data) // batch_size)
            val_progress_bar = tqdm(range(num_val_batches), desc="Validation batches")
            
            for val_batch_num in val_progress_bar:
                val_batch_idx = val_batch_num * batch_size
                val_batch = val_data[val_batch_idx:val_batch_idx + batch_size]
                
                try:
                    batch_data = prepare_sft_batch(val_batch, tokenizer)
                    input_ids = batch_data['input_ids']
                    attention_mask = batch_data['attention_mask']
                    labels = batch_data['labels']
                    
                    # Validation with BF16 mixed precision
                    with autocast(dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True
                        )
                    
                    val_loss += outputs.loss.item()
                    val_steps += 1
                    
                    val_progress_bar.set_postfix({
                        'val_loss': f'{val_loss / val_steps:.4f}'
                    })
                    
                except Exception as e:
                    print(f"\nError in validation batch {val_batch_idx}: {str(e)}")
                    continue
        
        model.train()
        
        # Calculate validation loss
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        
        # Log epoch metrics
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Training Loss: {avg_epoch_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Batches processed: {batch_count}")
        print(f"{'='*80}")
        
        # Log epoch metrics to W&B
        wandb.log({
            'epoch/train_loss': avg_epoch_train_loss,
            'epoch/val_loss': avg_val_loss,
            'epoch/epoch': epoch + 1,
            'epoch/batches': batch_count,
        })
        
        # Save checkpoint
        import os
        checkpoint_dir = f"results/checkpoints/epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ Checkpoint saved to {checkpoint_dir}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total global steps: {global_step}")
    print(f"Final model checkpoint: results/checkpoints/epoch_{num_epochs}")
