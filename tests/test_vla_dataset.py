"""Tests for VLA Dataset."""

import pytest
import torch
import json
import os
from vlafactory.data.vla_dataset import VLADataset


class TestVLADataset:
    """Test cases for VLADataset class."""
    
    def test_dummy_dataset_creation(self, tmp_path):
        """Test creating a dummy dataset."""
        save_path = tmp_path / "dummy_data.json"
        
        data_path = VLADataset.create_dummy_dataset(
            num_samples=50,
            save_path=str(save_path)
        )
        
        assert os.path.exists(data_path)
        
        # Load and verify
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 50
        assert 'instruction' in data[0]
        assert 'action' in data[0]
        
    def test_dataset_loading(self, tmp_path):
        """Test loading dataset from file."""
        # Create test data
        test_data = [
            {
                'instruction': 'Pick up the red block',
                'action': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
            {
                'instruction': 'Move to location A',
                'action': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
        ]
        
        data_path = tmp_path / "test_data.json"
        with open(data_path, 'w') as f:
            json.dump(test_data, f)
        
        # Load dataset
        dataset = VLADataset(
            data_path=str(data_path),
            action_dim=7
        )
        
        assert len(dataset) == 2
        
    def test_dataset_getitem(self, tmp_path):
        """Test getting items from dataset."""
        # Create test data
        test_data = [
            {
                'instruction': 'Test instruction',
                'action': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            }
        ]
        
        data_path = tmp_path / "test_data.json"
        with open(data_path, 'w') as f:
            json.dump(test_data, f)
        
        dataset = VLADataset(
            data_path=str(data_path),
            action_dim=7,
            max_length=512
        )
        
        # Get item
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'action_labels' in item
        assert item['action_labels'].shape == (7,)
        
    def test_action_padding(self, tmp_path):
        """Test that actions are padded correctly."""
        # Create test data with shorter action
        test_data = [
            {
                'instruction': 'Test',
                'action': [0.1, 0.2, 0.3]  # Only 3 dims
            }
        ]
        
        data_path = tmp_path / "test_data.json"
        with open(data_path, 'w') as f:
            json.dump(test_data, f)
        
        dataset = VLADataset(
            data_path=str(data_path),
            action_dim=7
        )
        
        item = dataset[0]
        
        # Should be padded to 7 dimensions
        assert item['action_labels'].shape == (7,)
        assert item['action_labels'][0] == pytest.approx(0.1)
        assert item['action_labels'][2] == pytest.approx(0.3)
        assert item['action_labels'][6] == pytest.approx(0.0)  # Padded
        
    def test_collate_fn(self, tmp_path):
        """Test batch collation."""
        # Create test data
        test_data = [
            {'instruction': 'Test 1', 'action': [0.1] * 7},
            {'instruction': 'Test 2', 'action': [0.2] * 7}
        ]
        
        data_path = tmp_path / "test_data.json"
        with open(data_path, 'w') as f:
            json.dump(test_data, f)
        
        dataset = VLADataset(
            data_path=str(data_path),
            action_dim=7
        )
        
        # Get batch
        batch = [dataset[0], dataset[1]]
        collated = VLADataset.collate_fn(batch)
        
        assert 'input_ids' in collated
        assert 'action_labels' in collated
        assert collated['action_labels'].shape[0] == 2  # Batch size
        
    def test_jsonl_loading(self, tmp_path):
        """Test loading JSONL format."""
        data_path = tmp_path / "test_data.jsonl"
        
        with open(data_path, 'w') as f:
            f.write(json.dumps({'instruction': 'Test 1', 'action': [0.1] * 7}) + '\n')
            f.write(json.dumps({'instruction': 'Test 2', 'action': [0.2] * 7}) + '\n')
        
        dataset = VLADataset(
            data_path=str(data_path),
            action_dim=7
        )
        
        assert len(dataset) == 2
