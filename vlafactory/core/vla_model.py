"""
VLA Model wrapper class.

This class wraps LlamaFactory's model architecture and extends it for 
Vision-Language-Action tasks.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model wrapper.
    
    This class extends language models from LlamaFactory to support
    vision-language-action learning by adding vision encoder and action head.
    
    Args:
        language_model: Base language model from LlamaFactory
        vision_encoder: Vision encoder for processing images
        action_dim: Dimension of action space
        hidden_dim: Hidden dimension for projection layers
    """
    
    def __init__(
        self,
        language_model: Optional[nn.Module] = None,
        vision_encoder: Optional[nn.Module] = None,
        action_dim: int = 7,
        hidden_dim: int = 768,
    ):
        super().__init__()
        
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Vision-language projection layer
        if vision_encoder is not None:
            self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.vision_proj = None
            
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLA model.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask for input
            images: Input images (if using vision encoder)
            labels: Language modeling labels
            action_labels: Action prediction labels
            **kwargs: Additional arguments for language model
            
        Returns:
            Dictionary containing loss and logits
        """
        outputs = {}
        
        # Process vision inputs if available
        vision_features = None
        if images is not None and self.vision_encoder is not None:
            vision_features = self.vision_encoder(images)
            vision_features = self.vision_proj(vision_features)
            
        # Process language inputs
        if self.language_model is not None:
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            
            if isinstance(lm_outputs, dict):
                outputs.update(lm_outputs)
                hidden_states = lm_outputs.get('hidden_states', lm_outputs.get('last_hidden_state'))
            else:
                # Handle tuple outputs from some models
                outputs['loss'] = lm_outputs[0] if labels is not None else None
                hidden_states = lm_outputs[1] if len(lm_outputs) > 1 else None
        else:
            # If no language model, use vision features directly
            hidden_states = vision_features
            
        # Predict actions
        if hidden_states is not None:
            # Use last token representation for action prediction
            if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                action_input = hidden_states[:, -1, :]
            else:
                action_input = hidden_states
                
            action_logits = self.action_head(action_input)
            outputs['action_logits'] = action_logits
            
            # Compute action loss if labels provided
            if action_labels is not None:
                action_loss = nn.functional.mse_loss(action_logits, action_labels)
                outputs['action_loss'] = action_loss
                
                # Combine losses if language loss exists
                if 'loss' in outputs and outputs['loss'] is not None:
                    outputs['loss'] = outputs['loss'] + action_loss
                else:
                    outputs['loss'] = action_loss
                    
        return outputs
    
    def generate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate actions for given inputs.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            images: Input images
            **kwargs: Additional arguments
            
        Returns:
            Predicted actions or None if no actions could be generated
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                **kwargs
            )
        return outputs.get('action_logits', None)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        action_dim: int = 7,
        hidden_dim: int = 768,
        **kwargs
    ) -> "VLAModel":
        """
        Load a pretrained VLA model.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded VLA model
        """
        # This is a placeholder - actual implementation would load from LlamaFactory
        # For now, create an empty model
        model = cls(
            language_model=None,
            vision_encoder=None,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        return model
    
    def save_pretrained(self, save_directory: str):
        """
        Save the VLA model.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config = {
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }
        
        import json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
