# VLAFactory

A minimal framework for training Vision-Language-Action (VLA) models, built on top of [Transformers](@TODO: add link here).

## Overview

VLAFactory extends LlamaFactory's capabilities to support training VLA models that combine:
- **Vision**: Processing visual observations from cameras
- **Language**: Understanding natural language instructions
- **Action**: Generating robot control actions

## Features

- 🎯 **Minimal Design**: Clean, focused components for VLA training
- 🔧 **LlamaFactory Integration**: Built on top of proven LLM training infrastructure
- 🤖 **Robot-Ready**: Support for continuous action spaces
- 📊 **Flexible Data**: Handle various VLA dataset formats
- 🚀 **Easy to Use**: Simple API for quick experimentation

## Installation

### From Source

```bash
git clone https://github.com/ECHO960/VLAFactory.git
cd VLAFactory
pip install -e .
```

### With Vision Support

```bash
pip install -e ".[vision]"
```

### With LlamaFactory

```bash
pip install -e ".[llamafactory]"
```

## Quick Start

### 1. Prepare Your Data

Create a JSON file with your VLA data:

```json
[
  {
    "instruction": "Pick up the red block and place it in the box",
    "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "image": "path/to/image.jpg"
  }
]
```

### 2. Train a Model

Using the example training script:

```bash
bash examples/train.sh
```

### 3. Use in Code

```python
from vlafactory import VLAModel, VLADataset, VLATrainer

# Create dataset
dataset = VLADataset(
    data_path="data/vla_data.json",
    action_dim=7,
    max_length=512
)

# Initialize model
model = VLAModel(
    action_dim=7,
    hidden_dim=768
)

# Train
trainer = VLATrainer(
    model=model,
    train_dataset=dataset,
    args={
        'num_train_epochs': 3,
        'per_device_train_batch_size': 8,
        'learning_rate': 5e-5,
    }
)

metrics = trainer.train()
trainer.save_model("./output")
```

## Architecture

### Core Components

1. **VLAModel** (`src/vlafactory/core/vla_model.py`)
   - Wraps language models from LlamaFactory
   - Adds vision encoder support
   - Implements action prediction head
   - Handles multimodal fusion

2. **VLADataset** (`src/vlafactory/data/vla_dataset.py`)
   - Loads and preprocesses VLA data
   - Supports JSON/JSONL formats
   - Handles images, text, and actions
   - Provides efficient batching

3. **VLATrainer** (`src/vlafactory/trainer/vla_trainer.py`)
   - Manages training loop
   - Compatible with LlamaFactory infrastructure
   - Supports evaluation and checkpointing
   - Handles mixed language and action losses

### Integration with LlamaFactory

VLAFactory is designed to work seamlessly with LlamaFactory:

```python
from llamafactory.model import load_model
from vlafactory import VLAModel

# Load base language model from LlamaFactory
language_model, tokenizer = load_model(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    finetuning_type="lora"
)

# Wrap with VLA capabilities
vla_model = VLAModel(
    language_model=language_model,
    action_dim=7,
    hidden_dim=768
)
```

## Configuration

VLAFactory uses a hierarchical configuration system:

```python
from vlafactory.configs import get_default_config, load_config

# Get default config
config = get_default_config()

# Or load from file
config = load_config("examples/qwen3_vl_8b_sft.yaml")
```

Example configuration:

```json
{
  "model": {
    "action_dim": 7,
    "hidden_dim": 768,
    "model_name_or_path": "meta-llama/Llama-2-7b-hf"
  },
  "data": {
    "data_path": "data/vla_data.json",
    "max_length": 512,
    "train_split": 0.9
  },
  "training": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-5,
    "output_dir": "./output"
  }
}
```

## Data Format

VLAFactory supports multiple data formats:

### JSON Format

```json
[
  {
    "instruction": "Natural language instruction",
    "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "image": "path/to/image.jpg",
    "response": "Optional response text"
  }
]
```

### JSONL Format

```jsonl
{"instruction": "Instruction 1", "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
{"instruction": "Instruction 2", "action": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
```

### Required Fields

- `instruction` or `text`: Natural language instruction
- `action`: List of action values (will be padded/truncated to `action_dim`)

### Optional Fields

- `image` or `image_path`: Path to image file
- `response`: Language response for multi-task learning

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=vlafactory --cov-report=html
```

## Examples

### Example 1: Basic Training

```python
from vlafactory import VLAModel, VLADataset, VLATrainer

# Load dataset
dataset = VLADataset(
    data_path="data/robot_tasks.json",
    action_dim=7
)

# Create model
model = VLAModel(action_dim=7, hidden_dim=768)

# Train
trainer = VLATrainer(model=model, train_dataset=dataset)
trainer.train()
```

### Example 2: With Vision

```python
from vlafactory import VLAModel, VLADataset, VLATrainer
from transformers import CLIPVisionModel, CLIPImageProcessor

# Load vision encoder
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create dataset with image processing
dataset = VLADataset(
    data_path="data/visual_tasks.json",
    image_processor=image_processor,
    action_dim=7
)

# Create model with vision
model = VLAModel(
    vision_encoder=vision_encoder,
    action_dim=7,
    hidden_dim=768
)

# Train
trainer = VLATrainer(model=model, train_dataset=dataset)
trainer.train()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0.

## Citation

If you use VLAFactory in your research, please cite:

```bibtex
@software{vlafactory2024,
  title = {VLAFactory: A Minimal Framework for Vision-Language-Action Models},
  author = {VLAFactory Team},
  year = {2024},
  url = {https://github.com/ECHO960/VLAFactory}
}
```

## Acknowledgments

- Built on top of [LlamaFactory](https://github.com/hiyouga/LlamaFactory)
- Inspired by recent work in VLA models and robotic learning

## Contact

For questions and feedback, please open an issue on GitHub.
