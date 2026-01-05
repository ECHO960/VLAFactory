#!/usr/bin/env python3
"""
Example script for training a VLA model using VLAFactory.

This demonstrates the basic usage of VLAFactory components.
"""

import argparse
import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlafactory import VLAModel, VLADataset, VLATrainer
from vlafactory.configs import get_default_config, load_config


def main():
    parser = argparse.ArgumentParser(description='Train a VLA model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to training data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--action_dim',
        type=int,
        default=7,
        help='Dimension of action space'
    )
    parser.add_argument(
        '--use_dummy_data',
        action='store_true',
        help='Use dummy data for testing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.num_epochs:
        config['training']['num_train_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
    if args.action_dim:
        config['model']['action_dim'] = args.action_dim
    
    print("=" * 60)
    print("VLAFactory Training")
    print("=" * 60)
    print(f"Action dimension: {config['model']['action_dim']}")
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"Number of epochs: {config['training']['num_train_epochs']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print("=" * 60)
    
    # Create or load dataset
    if args.use_dummy_data:
        print("\nCreating dummy dataset for testing...")
        data_path = VLADataset.create_dummy_dataset(
            num_samples=100,
            save_path=os.path.join(config['training']['output_dir'], 'dummy_data.json')
        )
        config['data']['data_path'] = data_path
        print(f"Dummy dataset created at: {data_path}")
    
    # Initialize dataset
    print(f"\nLoading dataset from: {config['data']['data_path']}")
    
    try:
        dataset = VLADataset(
            data_path=config['data']['data_path'],
            tokenizer=None,  # In real use, pass actual tokenizer from LlamaFactory
            max_length=config['data']['max_length'],
            action_dim=config['model']['action_dim'],
        )
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please provide a valid data path or use --use_dummy_data flag")
        return
    
    # Split into train and eval
    train_size = int(len(dataset) * config['data']['train_split'])
    eval_size = len(dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Initialize model
    print("\nInitializing VLA model...")
    model = VLAModel(
        language_model=None,  # In real use, load from LlamaFactory
        vision_encoder=None,  # Optional: add vision encoder
        action_dim=config['model']['action_dim'],
        hidden_dim=config['model']['hidden_dim'],
    )
    print("Model initialized")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = VLATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_size > 0 else None,
        tokenizer=None,
        args=config['training'],
    )
    
    # Train the model
    print("\nStarting training...")
    print("=" * 60)
    
    try:
        metrics = trainer.train()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final training loss: {metrics['train_loss']:.4f}")
        print(f"Total training steps: {metrics['num_steps']}")
        
        # Save the model
        print(f"\nSaving model to: {config['training']['output_dir']}")
        trainer.save_model(config['training']['output_dir'])
        
        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return
    

if __name__ == '__main__':
    main()
