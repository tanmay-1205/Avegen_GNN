# Pregnancy App GNN: User Segmentation & Nudge Recommendation

A Graph Neural Network system for pregnancy app user engagement prediction and personalized behavioral nudge recommendations.

## 🎯 Project Overview

This project implements a state-of-the-art GNN-based system that:
- **Classifies users** into 5 engagement segments (Low, Moderate, High, Health Focused, Content Consumer)
- **Recommends personalized nudges** from 5 behavioral intervention categories
- **Leverages graph relationships** between users, segments, and nudges for improved predictions
- **Provides interpretable results** with detailed segment and nudge descriptions

## 🏗️ Project Structure

```
Avegen_Gnn/
├── 📄 DOCUMENTATION.md          # Comprehensive 5-page technical documentation
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📄 LICENSE                   # MIT License
├── 📄 .gitignore               # Git ignore rules
│
├── 🧠 Core Implementation
│   ├── gnn_model_pyg.py        # GNN architecture with bi-interaction aggregation
│   ├── train_pyg.py            # Training pipeline with BPR loss
│   ├── predict_pyg.py          # Inference pipeline for segments & nudges
│   ├── data_preprocessing.py   # Feature engineering and normalization
│   └── graph_construction.py   # Graph building and edge weight calculation
│
├── 📊 Data & Models
│   ├── synthetic_pregnancy_app_data.csv  # Synthetic user behavior dataset
│   ├── best_model_pyg.pt       # Trained model weights
│   ├── training_pyg.log        # Training history and metrics
│   ├── graph_visualization.png # Graph structure visualization
│   └── preprocessed_data/      # Processed features and encoders
│
└── 📚 Documentation
    └── data_notions/
        └── preprocessed_dataframe_structure.txt  # Feature documentation
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd Avegen_Gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Preprocess data (if needed)
python -c "from data_preprocessing import DataPreprocessor; DataPreprocessor().preprocess()"

# Train model
python train_pyg.py

# Make predictions
python predict_pyg.py
```

### 3. Example Prediction Output
```python
{
    'user_segments': [
        {
            'segment': 'Health Focused',
            'probability': 0.85,
            'description': 'Users primarily interested in health tracking...'
        }
    ],
    'recommended_nudges': [
        {
            'nudge_type': 'NUDGE_2_PERSONALIZED_TIPS',
            'confidence': 0.92,
            'description': 'Custom health tips based on your current trimester...'
        }
    ]
}
```

## 🔬 Technical Highlights

### Graph Neural Network Architecture
- **Bi-Interaction Aggregation**: Captures both additive and multiplicative feature interactions
- **Multi-Task Learning**: Simultaneous segment classification and nudge recommendation
- **Attention Mechanism**: Feature-based edge weighting for improved relationships
- **PyTorch Geometric**: Efficient graph operations and message passing

### Key Features
- **5 User Segments**: Low/Moderate/High Engagement, Health Focused, Content Consumer
- **5 Nudge Categories**: Reminders, Progress Feedback, Tips, Goal Setting, Social Cues
- **Scalable Design**: Handles large user bases with efficient graph operations
- **Interpretable Results**: Clear descriptions for segments and nudge recommendations

### Model Performance
- **Architecture**: 128→64→32 hidden dimensions with bi-interaction convolution
- **Training**: BPR loss with negative sampling, Adam optimizer (lr=0.0005)
- **Regularization**: Dropout (0.2), gradient clipping, L2 regularization
- **Early Stopping**: Patience-based validation monitoring

## 📖 Documentation

For detailed technical documentation including:
- **Theoretical foundations** of GNNs and multi-task learning
- **Complete data pipeline** and preprocessing steps
- **Architecture details** and mathematical formulations
- **Training procedures** and hyperparameter configurations
- **Deployment guidelines** and future enhancements

👉 **See [DOCUMENTATION.md](DOCUMENTATION.md)** for the comprehensive 5-page technical guide.

## 🛠️ Key Components

### Core Files
- **`gnn_model_pyg.py`**: GNN model with bi-interaction convolution and multi-task heads
- **`train_pyg.py`**: Training loop with BPR loss and negative sampling
- **`predict_pyg.py`**: Inference pipeline with segment classification and nudge recommendation
- **`data_preprocessing.py`**: Feature engineering, normalization, and encoding
- **`graph_construction.py`**: Graph construction with user-segment-nudge relationships

### Data Pipeline
1. **Raw Data**: Synthetic pregnancy app user behavior (1000+ users, 40+ features)
2. **Preprocessing**: Missing value handling, outlier treatment, feature engineering
3. **Graph Construction**: User-segment-nudge heterogeneous graph with attention weights
4. **Training**: BPR loss optimization with negative sampling
5. **Prediction**: Multi-task inference for segments and nudges

## 🔮 Future Enhancements

### Technical
- **Temporal GNNs**: Time-aware user behavior modeling
- **Heterogeneous Message Passing**: Different aggregation for different node types
- **Dynamic Graphs**: Real-time edge weight updates
- **Federated Learning**: Privacy-preserving distributed training

### Business
- **A/B Testing**: Nudge effectiveness measurement
- **Real-time Personalization**: Dynamic recommendation updates
- **Multi-modal Features**: Text, image, and sensor data integration
- **Causal Inference**: Understanding intervention effectiveness

## 📊 Results

The current model demonstrates:
- **Effective Segmentation**: Clear user categorization into meaningful engagement groups
- **Relevant Nudges**: Contextual behavioral intervention recommendations
- **Scalable Architecture**: Efficient handling of large user bases
- **Interpretable Output**: Actionable insights for product teams

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out to the development team.

---

**Built with ❤️ for improving pregnancy health outcomes through intelligent user engagement** 