import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from gnn_model import PregnancyGNN
from data_preprocessing import PregnancyDataPreprocessor
from graph_construction import GraphConstructor

class TrainingEngine:
    def __init__(self, 
                 model,
                 learning_rate=0.001,
                 weight_decay=5e-4,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the training engine
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = Adam(model.parameters(), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'segment_f1': [],
            'nudge_f1': []
        }
        
    def train_epoch(self, data):
        """
        Train for one epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Forward pass
        segment_preds, nudge_preds = self.model(x, edge_index)
        
        # Calculate loss for segments (only for user nodes)
        segment_loss = F.cross_entropy(
            segment_preds[:1000],  # Only user nodes
            data.segment_labels[:1000].to(self.device)
        )
        
        # Calculate loss for nudges (only for user nodes)
        nudge_loss = F.cross_entropy(
            nudge_preds[:1000],  # Only user nodes
            data.nudge_labels[:1000].to(self.device)
        )
        
        # Combined loss
        loss = segment_loss + nudge_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """
        Evaluate the model
        """
        self.model.eval()
        with torch.no_grad():
            # Move data to device
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            
            # Forward pass
            segment_preds, nudge_preds = self.model(x, edge_index)
            
            # Get predictions for user nodes only
            segment_preds = segment_preds[:1000].cpu().numpy()
            nudge_preds = nudge_preds[:1000].cpu().numpy()
            
            # Convert to class predictions
            segment_pred_classes = np.argmax(segment_preds, axis=1)
            nudge_pred_classes = np.argmax(nudge_preds, axis=1)
            
            # Calculate metrics
            segment_f1 = f1_score(
                data.segment_labels[:1000].cpu().numpy(),
                segment_pred_classes,
                average='weighted'
            )
            
            nudge_f1 = f1_score(
                data.nudge_labels[:1000].cpu().numpy(),
                nudge_pred_classes,
                average='weighted'
            )
            
            return segment_f1, nudge_f1
    
    def train(self, data, num_epochs=100, eval_every=10):
        """
        Full training loop
        """
        print(f"Training on device: {self.device}")
        
        best_f1 = 0.0
        pbar = tqdm(range(num_epochs), desc="Training")
        
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(data)
            self.history['train_loss'].append(train_loss)
            
            # Evaluation
            if (epoch + 1) % eval_every == 0:
                segment_f1, nudge_f1 = self.evaluate(data)
                self.history['segment_f1'].append(segment_f1)
                self.history['nudge_f1'].append(nudge_f1)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{train_loss:.4f}",
                    'segment_f1': f"{segment_f1:.4f}",
                    'nudge_f1': f"{nudge_f1:.4f}"
                })
                
                # Save best model
                if segment_f1 + nudge_f1 > best_f1:
                    best_f1 = segment_f1 + nudge_f1
                    torch.save(self.model.state_dict(), 'best_model.pt')
    
    def plot_training_history(self):
        """
        Plot training metrics
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(1, 2, 2)
        plt.plot(self.history['segment_f1'], label='Segment F1')
        plt.plot(self.history['nudge_f1'], label='Nudge F1')
        plt.title('F1 Scores')
        plt.xlabel('Evaluation Step')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('synthetic_pregnancy_app_data.csv')
    preprocessor = PregnancyDataPreprocessor()
    df_cleaned, df_engagement, df_health = preprocessor.fit_transform(df)
    
    # Create graph
    print("Constructing graph...")
    graph_constructor = GraphConstructor()
    graph_data = graph_constructor.create_graph(df_cleaned, df_engagement, df_health)
    
    # Add synthetic labels for testing
    # In real implementation, these would come from your data
    graph_data.segment_labels = torch.randint(0, 6, (len(graph_data.x),))
    graph_data.nudge_labels = torch.randint(0, 5, (len(graph_data.x),))
    
    # Initialize model
    print("Initializing model...")
    input_dim = graph_data.x.size(1)
    model = PregnancyGNN(input_dim=input_dim)
    
    # Initialize training engine
    trainer = TrainingEngine(model)
    
    # Train model
    print("Starting training...")
    trainer.train(graph_data)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("Training complete! Best model saved as 'best_model.pt'")