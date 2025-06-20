# Pregnancy App GNN: Complete Pipeline Documentation

## Table of Contents
1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Data Pipeline and Preprocessing](#3-data-pipeline-and-preprocessing)
4. [Graph Neural Network Architecture](#4-graph-neural-network-architecture)
5. [Training and Prediction Pipeline](#5-training-and-prediction-pipeline)

---

## 1. Introduction and Problem Statement

### 1.1 Project Overview
This project implements a Graph Neural Network (GNN) system for pregnancy app user engagement prediction and personalized nudge recommendations. The system analyzes user behavior patterns to classify users into engagement segments and recommend appropriate behavioral interventions.

### 1.2 Business Problem
Pregnancy apps face challenges in:
- **User Retention**: High dropout rates during pregnancy journey
- **Engagement Optimization**: Difficulty in identifying user preferences and needs
- **Personalization**: Generic recommendations that don't match individual user profiles
- **Behavioral Intervention**: Lack of targeted nudges to improve health outcomes

### 1.3 Solution Approach
We address these challenges through:
- **User Segmentation**: Classify users into 5 distinct engagement profiles
- **Nudge Recommendation**: Predict optimal behavioral interventions from 5 nudge categories
- **Graph-based Learning**: Leverage relationships between users, segments, and nudges
- **Multi-task Prediction**: Simultaneous segment classification and nudge recommendation

### 1.4 Key Features
- **Scalable Architecture**: Handles large user bases with efficient graph operations
- **Real-time Predictions**: Fast inference for production deployment
- **Interpretable Results**: Clear segment descriptions and nudge explanations
- **Extensible Design**: Easy addition of new segments or nudge types

---

## 2. Theoretical Foundations

### 2.1 Graph Neural Networks (GNNs)

#### 2.1.1 Mathematical Foundation
Graph Neural Networks operate on graph-structured data G = (V, E, X), where:
- V: Set of nodes (users, segments, nudges)
- E: Set of edges (relationships between nodes)
- X: Node feature matrix

The core GNN operation is message passing:
```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

Where:
- h_v^(l): Node v's representation at layer l
- N(v): Neighbors of node v
- UPDATE and AGGREGATE: Learnable functions

#### 2.1.2 Bi-Interaction Aggregation
Our model uses bi-interaction aggregation to capture complex user-segment relationships:

```
h_v^(l+1) = σ(W_1(h_v^(l) + h_neighbors^(l)) + W_2(h_v^(l) ⊙ h_neighbors^(l)))
```

Where:
- ⊙: Element-wise multiplication
- σ: Activation function (LeakyReLU)
- W_1, W_2: Learnable weight matrices

This captures both additive and multiplicative interactions between node features.

### 2.2 Multi-Task Learning Framework

#### 2.2.1 Shared Representation Learning
The model learns shared representations through GNN layers, then uses task-specific heads:

```
z_user = GNN(X, A)  # Shared user embeddings
y_segment = Segment_Head(z_user)  # Segment predictions
y_nudge = Nudge_Head(z_user)      # Nudge predictions
```

#### 2.2.2 Loss Function
The training objective combines segment classification and nudge recommendation:

```
L_total = L_BPR + λ_reg * L_regularization
```

Where L_BPR is Bayesian Personalized Ranking loss:
```
L_BPR = -∑ ln(σ(r_ui+ - r_ui-))
```

### 2.3 Attention Mechanism
Feature-based attention weights edges based on user-segment compatibility:

```
α_ij = softmax(LeakyReLU(a^T[W_h h_i || W_h h_j]))
```

Where:
- α_ij: Attention weight between nodes i and j
- a: Learnable attention vector
- ||: Concatenation operation

---

## 3. Data Pipeline and Preprocessing

### 3.1 Data Sources and Structure

#### 3.1.1 Raw Data Schema
The synthetic pregnancy app dataset contains:
- **User Demographics**: Age, city, occupation, pregnancy history
- **Engagement Metrics**: Session counts, content interactions, feature usage
- **Health Data**: BMI, health tracking frequency, medical conditions
- **Temporal Features**: Session patterns, notification responses

#### 3.1.2 Feature Categories
1. **Engagement Features** (13 metrics):
   - `session_start_count`: App launch frequency
   - `watch_now_count`: Video content consumption
   - `community_opened_count`: Social feature usage
   - `health_tracking_submission_count`: Health data logging
   - Additional interaction counters for various app features

2. **Health Features** (4 metrics):
   - `bmi`: Body Mass Index
   - `selected_height`, `selected_weight`: Physical measurements
   - `health_engagement_score`: Composite health activity score

3. **Derived Features** (8 metrics):
   - `total_engagement_score`: Composite engagement metric
   - `content_interaction_ratio`: Content usage relative to sessions
   - `community_engagement_ratio`: Social interaction frequency
   - `health_tracking_ratio`: Health logging consistency

### 3.2 Preprocessing Pipeline

#### 3.2.1 Data Cleaning and Validation
```python
class DataPreprocessor:
    def handle_missing_values(self):
        # Numerical: Median imputation
        # Categorical: Mode imputation
        
    def handle_outliers(self):
        # IQR method for height/weight
        # Clipping to [Q1-1.5*IQR, Q3+1.5*IQR]
```

#### 3.2.2 Feature Engineering
1. **BMI Calculation**: BMI = weight(kg) / height(m)²
2. **Age Groups**: Binning into pregnancy-relevant categories
3. **Engagement Ratios**: Normalized interaction rates
4. **Temporal Features**: Days since last activity

#### 3.2.3 Normalization and Encoding
- **StandardScaler**: Zero-mean, unit-variance for numerical features
- **LabelEncoder**: Categorical to numerical mapping
- **Feature Scaling**: Ensures equal contribution across feature types

### 3.3 Graph Construction

#### 3.3.1 Node Types
1. **User Nodes**: Individual app users (n_users)
2. **Segment Nodes**: 5 engagement categories
3. **Nudge Nodes**: 5 behavioral intervention types

#### 3.3.2 Edge Construction
- **User-Segment Edges**: Based on feature similarity
- **Segment-Nudge Edges**: Predefined intervention mappings
- **Edge Weights**: Attention-based compatibility scores

---

## 4. Graph Neural Network Architecture

### 4.1 Model Architecture Overview

#### 4.1.1 Core Components
```python
class PregnancyAppGNN_PyG(nn.Module):
    def __init__(self, args, n_users, n_segments, n_nudges, n_feature_types):
        # Feature encoder: Raw features → embeddings
        self.feature_encoder = nn.Sequential(...)
        
        # Node embeddings
        self.segment_embed = nn.Embedding(n_segments, embed_dim)
        self.nudge_embed = nn.Embedding(n_nudges, embed_dim)
        
        # GNN layers
        self.convs = nn.ModuleList([...])
        
        # Prediction heads
        self.segment_predictor = nn.Sequential(...)
        self.nudge_predictor = nn.Sequential(...)
```

#### 4.1.2 Layer Configuration
- **Input Dimension**: Number of user features (varies by preprocessing)
- **Hidden Dimensions**: [128, 64, 32] (configurable)
- **Embedding Dimension**: 128
- **Dropout Rates**: [0.2, 0.2, 0.2] for regularization

### 4.2 Forward Pass Mechanics

#### 4.2.1 Training Mode
1. **Feature Encoding**: Transform raw user features to embeddings
2. **Node Concatenation**: Combine user, segment, and nudge embeddings
3. **Message Passing**: Apply GNN layers with bi-interaction aggregation
4. **Loss Calculation**: BPR loss with negative sampling

#### 4.2.2 Prediction Mode
1. **Feature Encoding**: Same as training
2. **Representation Learning**: Generate user embeddings through GNN
3. **Multi-task Prediction**: 
   - Segment classification via segment_predictor
   - Nudge recommendation via nudge_predictor

### 4.3 Bi-Interaction Convolution

#### 4.3.1 Implementation
```python
class BiInteractionConv(MessagePassing):
    def message(self, x_i, x_j, edge_weight):
        sum_embed = self.lin1(x_i + x_j)      # Additive interaction
        bi_embed = self.lin2(x_i * x_j)       # Multiplicative interaction
        return (sum_embed + bi_embed) * edge_weight
```

#### 4.3.2 Advantages
- **Rich Interactions**: Captures both linear and non-linear relationships
- **Feature Crossing**: Automatic feature interaction discovery
- **Scalability**: Efficient computation through message passing

### 4.4 Attention Mechanism

#### 4.4.1 Feature-Based Attention
Weights edges based on feature compatibility:
```python
def update_attention(self, user_list, segment_list, feature_type_list):
    # Calculate attention scores per feature type
    # Normalize using softmax
    # Create weighted edge attributes
```

#### 4.4.2 Dynamic Graph Updates
- **Training**: Attention weights updated based on learning
- **Inference**: Fixed attention for consistent predictions

---

## 5. Training and Prediction Pipeline

### 5.1 Training Configuration

#### 5.1.1 Hyperparameters
```python
# Model Architecture
embed_dim = 128
conv_dim_list = [128, 64, 32]
mess_dropout = [0.2, 0.2, 0.2]
aggregation_type = 'bi-interaction'

# Training Parameters
learning_rate = 0.0005
batch_size = 128
n_epochs = 100
n_neg_samples = 5

# Segments and Nudges
n_segments = 5
n_nudges = 5
```

#### 5.1.2 Optimization Strategy
- **Optimizer**: Adam with L2 regularization
- **Learning Rate**: 0.0005 (balanced convergence)
- **Gradient Clipping**: Max norm 1.0 (numerical stability)
- **Early Stopping**: Patience 10 epochs

### 5.2 Training Process

#### 5.2.1 Data Splitting
- **Training Set**: 80% of user-segment interactions
- **Validation Set**: 20% for early stopping
- **Negative Sampling**: 5 negative samples per positive interaction

#### 5.2.2 Loss Computation
```python
def _forward_train(self, data):
    # 1. Encode user features
    user_embed = self.feature_encoder(data.x[:self.n_users])
    
    # 2. Message passing through GNN layers
    x = self.apply_gnn_layers(combined_embeddings)
    
    # 3. Calculate BPR loss with negative sampling
    loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
    
    # 4. Add L2 regularization
    return loss + l2_regularization
```

### 5.3 Prediction Pipeline

#### 5.3.1 User Segmentation
```python
def get_segments(self, segment_scores, threshold=0.3):
    segment_probs = torch.softmax(segment_scores, dim=0)
    return [segments above threshold with descriptions]
```

**Segment Categories**:
1. **Low Engagement**: Minimal app activity, needs motivation
2. **Moderate Engagement**: Average usage, targeted content candidates
3. **High Engagement**: Active users, advanced features suitable
4. **Health Focused**: Prioritizes health tracking and medical info
5. **Content Consumer**: High educational content consumption

#### 5.3.2 Nudge Recommendation
```python
def get_top_nudges(self, nudge_scores, k=3):
    nudge_probs = torch.softmax(nudge_scores, dim=0)
    return [top-k nudges with confidence scores]
```

**Nudge Categories**:
1. **Reminder and Prompts**: Gentle activity reminders
2. **Progress Feedback**: Personalized journey updates
3. **Personalized Tips**: Health advice based on trimester
4. **Goal Settings**: Achievable target recommendations
5. **Social Cues**: Community engagement prompts

### 5.4 Model Evaluation and Deployment

#### 5.4.1 Performance Metrics
- **Training Loss**: BPR loss convergence
- **Validation Loss**: Generalization capability
- **Segment Accuracy**: Classification performance
- **Nudge Relevance**: Recommendation quality

#### 5.4.2 Production Deployment
```python
# Load trained model
predictor = EngagementPredictor('best_model_pyg.pt')

# Make predictions
predictions = predictor.predict(user_data)

# Extract results
segments = predictions['user_segments']
nudges = predictions['recommended_nudges']
```

### 5.5 Key Files and Usage

#### 5.5.1 Core Implementation Files
- **`gnn_model_pyg.py`**: GNN architecture and forward pass logic
- **`train_pyg.py`**: Training pipeline and hyperparameter configuration
- **`predict_pyg.py`**: Inference pipeline and result interpretation
- **`data_preprocessing.py`**: Data cleaning and feature engineering
- **`graph_construction.py`**: Graph building and edge weight calculation

#### 5.5.2 Usage Commands
```bash
# 1. Preprocess data
python -c "from data_preprocessing import DataPreprocessor; DataPreprocessor().preprocess()"

# 2. Train model
python train_pyg.py

# 3. Make predictions
python predict_pyg.py
```

#### 5.5.3 Model Artifacts
- **`best_model_pyg.pt`**: Trained model weights
- **`preprocessed_data/`**: Processed features and encoders
- **`training_pyg.log`**: Training history and metrics

### 5.6 Future Enhancements

#### 5.6.1 Technical Improvements
- **Dynamic Graph Updates**: Real-time edge weight adaptation
- **Heterogeneous GNNs**: Different message passing for different node types
- **Temporal Modeling**: Time-aware user behavior prediction
- **Federated Learning**: Privacy-preserving distributed training

#### 5.6.2 Business Extensions
- **A/B Testing Integration**: Nudge effectiveness measurement
- **Real-time Personalization**: Dynamic recommendation updates
- **Multi-modal Features**: Integration of text, image, and sensor data
- **Causal Inference**: Understanding intervention effectiveness

---

This documentation provides a comprehensive overview of the pregnancy app GNN system, from theoretical foundations to practical implementation details. The modular design ensures maintainability and extensibility for future enhancements. 