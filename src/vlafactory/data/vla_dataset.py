"""
VLA Dataset class for handling vision-language-action data.
"""

from typing import Dict, Any, Optional, List, Union
import torch
from torch.utils.data import Dataset
import json
import os


class VLADataset(Dataset):
    """
    Dataset for Vision-Language-Action learning.
    
    This dataset handles multimodal data including:
    - Text instructions/descriptions
    - Visual observations (images)
    - Action labels
    
    Args:
        data_path: Path to dataset file or directory
        tokenizer: Tokenizer for processing text
        image_processor: Processor for images (optional)
        max_length: Maximum sequence length for text
        action_dim: Dimension of action space
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        max_length: int = 512,
        action_dim: int = 7,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.action_dim = action_dim
        
        # Load data
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            List of data samples
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist")
            
        # Handle JSON file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        # Handle JSONL file
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            # Assume it's a directory with data files
            data = []
            if os.path.isdir(data_path):
                for file in os.listdir(data_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(data_path, file)
                        with open(file_path, 'r') as f:
                            data.append(json.load(f))
                            
        return data if isinstance(data, list) else [data]
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing processed inputs
        """
        sample = self.data[idx]
        
        # Process text instruction
        instruction = sample.get('instruction', sample.get('text', ''))
        
        processed = {}
        
        # Tokenize text
        if self.tokenizer is not None and instruction:
            tokenized = self.tokenizer(
                instruction,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            processed['input_ids'] = tokenized['input_ids'].squeeze(0)
            processed['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        else:
            # Return dummy tokens if no tokenizer
            processed['input_ids'] = torch.zeros(self.max_length, dtype=torch.long)
            processed['attention_mask'] = torch.ones(self.max_length, dtype=torch.long)
        
        # Process image if available
        image_path = sample.get('image', sample.get('image_path'))
        if image_path and self.image_processor is not None:
            # Load and process image
            try:
                from PIL import Image
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    processed_image = self.image_processor(image, return_tensors='pt')
                    processed['images'] = processed_image['pixel_values'].squeeze(0)
            except Exception as e:
                # If image processing fails, skip image
                processed['images'] = None
        else:
            processed['images'] = None
            
        # Process action labels
        if 'action' in sample:
            action = sample['action']
            if isinstance(action, list):
                action_tensor = torch.tensor(action, dtype=torch.float32)
            else:
                action_tensor = torch.tensor([action], dtype=torch.float32)
                
            # Pad or truncate to action_dim
            if len(action_tensor) < self.action_dim:
                padding = torch.zeros(self.action_dim - len(action_tensor))
                action_tensor = torch.cat([action_tensor, padding])
            elif len(action_tensor) > self.action_dim:
                action_tensor = action_tensor[:self.action_dim]
                
            processed['action_labels'] = action_tensor
        
        # Add any additional fields
        if 'response' in sample:
            processed['labels'] = processed['input_ids'].clone()
            
        return processed
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching samples.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dictionary
        """
        collated = {}
        
        # Collect all keys
        keys = set()
        for sample in batch:
            keys.update(sample.keys())
        
        # Batch each field
        for key in keys:
            values = [sample.get(key) for sample in batch if sample.get(key) is not None]
            
            if not values:
                continue
                
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
                
        return collated
