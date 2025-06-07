# GNN-Based Health Engagement System

A Graph Neural Network (GNN) based system for personalizing health engagement nudges in the Avegen HealthMachine platform.

## Project Overview

This project implements an AI-driven engagement system using Graph Neural Networks to improve patient engagement through personalized nudges. The system learns from user interactions, health metrics, and engagement patterns to recommend the most effective engagement strategies.

### Key Features

- GNN-based user-segment-nudge relationship modeling
- Dynamic user engagement tracking
- Personalized nudge recommendations
- Inductive learning capabilities for new users
- Sliding window training for temporal adaptation

## Architecture

The system uses a multi-layer GNN with:
- User nodes (engagement + health features)
- Segment nodes (user groupings)
- Nudge nodes (engagement actions)
- Message passing layers with Xavier initialization
- Attention mechanisms for feature weighting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/avegen-gnn.git
cd avegen-gnn

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
avegen-gnn/
├── data/
│   ├── preprocessing.py
│   └── simulation.py
├── models/
│   ├── gnn.py
│   └── layers.py
├── training/
│   ├── trainer.py
│   └── evaluation.py
├── utils/
│   └── graph_construction.py
├── requirements.txt
└── README.md
```

## Usage

```python
from models.gnn import GNN
from training.trainer import train_model

# Initialize model
model = GNN(
    user_dim=6,
    segment_dim=10,
    nudge_dim=15,
    hidden_dim=64
)

# Train model
train_model(model, train_data, val_data)
```

## Model Details

The GNN architecture includes:
- Input Layer: Processes user (6D), segment, and nudge features
- Message Passing Layers: Transform and propagate node information
- Xavier Initialization: Optimized weight initialization
- Output Layer: Generates engagement predictions

## Training Approaches

1. **Full Training**
   - Complete model training on all available data
   - Used for initial model setup

2. **Sliding Window**
   - Maintains recent data window
   - Updates model with new user interactions

3. **Inductive Learning**
   - Handles new users and segments
   - Adapts to changing engagement patterns

## Results

- Improved user engagement metrics
- Better nudge targeting
- Dynamic adaptation to user behavior

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/avegen-gnn 