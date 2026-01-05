"""
VLA Trainer class that extends LlamaFactory's training capabilities.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
import os


class VLATrainer:
    """
    Trainer for Vision-Language-Action models.
    
    This class provides training loop and utilities for VLA models,
    designed to work with LlamaFactory's infrastructure.
    
    Args:
        model: VLA model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        tokenizer: Tokenizer for the model
        args: Training arguments
    """
    
    def __init__(
        self,
        model: Any,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        args: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.args = args or {}
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model is not None:
            self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = None
        self.scheduler = None
        
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if self.optimizer is None and self.model is not None:
            learning_rate = self.args.get('learning_rate', 5e-5)
            weight_decay = self.args.get('weight_decay', 0.01)
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            
            # Setup scheduler if specified
            if self.args.get('lr_scheduler_type'):
                num_training_steps = self.args.get('num_training_steps', 1000)
                warmup_steps = self.args.get('warmup_steps', 100)
                
                from torch.optim.lr_scheduler import LinearLR, SequentialLR
                
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=warmup_steps
                )
                
                self.scheduler = warmup_scheduler
    
    def train(self) -> Dict[str, float]:
        """
        Execute training loop.
        
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None or self.train_dataset is None:
            raise ValueError("Model and train_dataset must be provided")
        
        self._setup_optimizer()
        
        # Setup dataloader
        batch_size = self.args.get('per_device_train_batch_size', 8)
        num_epochs = self.args.get('num_train_epochs', 3)
        
        # Get collate function from dataset if available
        # Handle both regular datasets and Subset from random_split
        dataset = self.train_dataset
        if hasattr(dataset, 'dataset'):
            # This is a Subset, get the underlying dataset
            dataset = dataset.dataset
        collate_fn = getattr(dataset, 'collate_fn', None)
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get('loss')
                
                if loss is None:
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                max_grad_norm = self.args.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                
                self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Track metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_steps += 1
                
                # Logging
                if batch_idx % self.args.get('logging_steps', 10) == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Step {batch_idx}/{len(train_dataloader)}, "
                          f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if self.eval_dataset is not None and (epoch + 1) % self.args.get('eval_epochs', 1) == 0:
                eval_metrics = self.evaluate()
                print(f"Evaluation metrics: {eval_metrics}")
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'num_steps': num_steps,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None or self.eval_dataset is None:
            return {}
        
        self.model.eval()
        
        batch_size = self.args.get('per_device_eval_batch_size', 8)
        
        # Handle both regular datasets and Subset from random_split
        dataset = self.eval_dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        collate_fn = getattr(dataset, 'collate_fn', None)
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.get('loss')
                
                if loss is not None:
                    total_loss += loss.item() * len(batch.get('input_ids', [1]))
                    num_samples += len(batch.get('input_ids', [1]))
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'num_samples': num_samples,
        }
    
    def save_model(self, output_dir: str):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
            torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading")
        
        # Check if model class has from_pretrained class method
        if hasattr(self.model.__class__, 'from_pretrained'):
            self.model = self.model.__class__.from_pretrained(model_path)
        else:
            state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")
