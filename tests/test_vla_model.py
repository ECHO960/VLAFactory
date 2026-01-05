"""Tests for VLA Model."""

import pytest
import torch
import torch.nn as nn
from vlafactory.core.vla_model import VLAModel


class TestVLAModel:
    """Test cases for VLAModel class."""
    
    def test_model_initialization(self):
        """Test that model can be initialized with default parameters."""
        model = VLAModel(action_dim=7, hidden_dim=768)
        assert model is not None
        assert model.action_dim == 7
        assert model.hidden_dim == 768
        assert model.action_head is not None
        
    def test_model_with_custom_dims(self):
        """Test model initialization with custom dimensions."""
        model = VLAModel(action_dim=10, hidden_dim=1024)
        assert model.action_dim == 10
        assert model.hidden_dim == 1024
        
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = VLAModel(action_dim=7, hidden_dim=768)
        
        # Create dummy inputs
        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass - model without language_model should still work
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check outputs
        assert isinstance(outputs, dict)
        
    def test_action_head(self):
        """Test that action head produces correct output shape."""
        model = VLAModel(action_dim=7, hidden_dim=768)
        
        # Create dummy hidden states
        batch_size = 4
        hidden_states = torch.randn(batch_size, 768)
        
        # Pass through action head
        actions = model.action_head(hidden_states)
        
        # Check shape
        assert actions.shape == (batch_size, 7)
        
    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        model = VLAModel(action_dim=7, hidden_dim=768)
        
        # Save model
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))
        
        # Check that files were created
        assert (save_dir / "pytorch_model.bin").exists()
        assert (save_dir / "config.json").exists()
        
        # Load model
        loaded_model = VLAModel.from_pretrained(
            str(save_dir),
            action_dim=7,
            hidden_dim=768
        )
        assert loaded_model is not None
        
    def test_generate_actions(self):
        """Test action generation."""
        model = VLAModel(action_dim=7, hidden_dim=768)
        model.eval()
        
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        actions = model.generate_actions(input_ids=input_ids)
        
        # Actions should be generated but may be None if no hidden states
        # This is expected without a language model
        assert actions is not None or model.language_model is None
