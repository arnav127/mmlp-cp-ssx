"""
Module 2: Deep Temporal Base Model & Hidden State Extractor
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Implements a PyTorch Multi-Task Bi-LSTM sequence architecture that processes weekly
clickstream data and outputs joint predictions for Dropout Risk P(Y) and Final GPA.
Uses LayerNorm for batch-size invariant bottleneck representations h_{i,t} in R^32.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score

class BiLSTMStudentModel(nn.Module):
    """
    Multi-Task Bidirectional LSTM neural network for temporal clickstream modeling.
    Extracts a 32-dimensional bottleneck hidden state representation h_{i,t}.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, bottleneck_dim: int = 32):
        super(BiLSTMStudentModel, self).__init__()
        
        # Bidirectional LSTM Layer (outputs 2 * hidden_dim = 128)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # LayerNorm Projection Hook Layer (128 -> 32)
        # LayerNorm ensures batch-size invariant embeddings for single student inference
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-Task Classification & Regression Heads
        self.head_dropout = nn.Linear(bottleneck_dim, 1)  # Dropout risk P(Y)
        self.head_gpa = nn.Linear(bottleneck_dim, 1)      # Expected Final GPA (0 to 1 scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Bi-LSTM, bottleneck projection, and multi-task heads.
        
        Parameters:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - risk_prob: Dropout risk probabilities P(Y) in [0, 1]
                - gpa_pred: Expected final GPA in [1.0, 4.0]
                - latent_h: Bottleneck latent representations h_{i,t} in R^32
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.bilstm(x)
        
        # Take the hidden representation at the final time step t = T
        last_step_out = lstm_out[:, -1, :]  # (batch_size, 128)
        
        # Bottleneck hidden representation h_{i,t} in R^32
        latent_h = self.bottleneck(last_step_out)
        
        # Multi-task heads
        risk_logit = self.head_dropout(latent_h)
        risk_prob = torch.sigmoid(risk_logit)
        
        gpa_norm_pred = torch.sigmoid(self.head_gpa(latent_h))
        gpa_pred = 1.0 + 3.0 * gpa_norm_pred  # Map to [1.0, 4.0] GPA scale
        
        return risk_prob.squeeze(-1), gpa_pred.squeeze(-1), latent_h

    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Intercepts and returns bottleneck latent hidden vectors h_{i,t} in R^32.
        """
        self.eval()
        with torch.no_grad():
            _, _, latent_h = self.forward(x)
        return latent_h

def train_base_lstm(
    clickstream_tensor: np.ndarray,
    static_df: pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.003,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[BiLSTMStudentModel, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Trains the multi-task Bi-LSTM base model using standardized input features and joint loss.
    """
    num_students, seq_len, num_features = clickstream_tensor.shape
    
    # split data first to compute mean/std only on train data
    indices = np.arange(num_students)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=static_df["dropout"].values
    )
    
    train_clickstream = clickstream_tensor[train_idx]
    
    # 1. Standardize 3D clickstream tensor per feature across time and samples
    flattened_train = train_clickstream.reshape(-1, num_features)
    mean_x = np.mean(flattened_train, axis=0)
    std_x = np.std(flattened_train, axis=0) + 1e-6
    norm_clickstream = (clickstream_tensor - mean_x) / std_x

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(norm_clickstream, dtype=torch.float32)
    y_dropout = torch.tensor(static_df["dropout"].values, dtype=torch.float32)
    y_gpa = torch.tensor(static_df["final_gpa"].values, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_tensor[train_idx], y_dropout[train_idx], y_gpa[train_idx])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = BiLSTMStudentModel(input_dim=num_features, hidden_dim=64, bottleneck_dim=32)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_x, b_y_drop, b_y_gpa in dataloader:
            optimizer.zero_grad()
            pred_drop, pred_gpa, _ = model(b_x)
            
            # Loss heads operating on normalized scales
            loss_drop = bce_loss_fn(pred_drop, b_y_drop)
            loss_gpa = mse_loss_fn((pred_gpa - 1.0)/3.0, (b_y_gpa - 1.0)/3.0)
            
            joint_loss = 0.6 * loss_drop + 0.4 * loss_gpa
            joint_loss.backward()
            optimizer.step()
            epoch_loss += joint_loss.item()
            
    # Extract predictions and latent representations across full dataset
    model.eval()
    with torch.no_grad():
        full_drop_prob, full_gpa_pred, latent_h_matrix = model(X_tensor)
        
    predictions = {
        "pred_dropout_risk": full_drop_prob.numpy(),
        "pred_final_gpa": full_gpa_pred.numpy()
    }
    
    # Compute diagnostics
    train_drop_prob = full_drop_prob[train_idx].numpy()
    test_drop_prob = full_drop_prob[test_idx].numpy()
    train_gpa_pred = full_gpa_pred[train_idx].numpy()
    test_gpa_pred = full_gpa_pred[test_idx].numpy()
    
    y_drop_train = y_dropout[train_idx].numpy()
    y_drop_test = y_dropout[test_idx].numpy()
    y_gpa_train = y_gpa[train_idx].numpy()
    y_gpa_test = y_gpa[test_idx].numpy()
    
    diagnostics = {
        'train_auc': float(roc_auc_score(y_drop_train, train_drop_prob)),
        'test_auc': float(roc_auc_score(y_drop_test, test_drop_prob)),
        'train_accuracy': float(accuracy_score(y_drop_train, (train_drop_prob >= 0.5).astype(int))),
        'test_accuracy': float(accuracy_score(y_drop_test, (test_drop_prob >= 0.5).astype(int))),
        'train_f1': float(f1_score(y_drop_train, (train_drop_prob >= 0.5).astype(int))),
        'test_f1': float(f1_score(y_drop_test, (test_drop_prob >= 0.5).astype(int))),
        'train_gpa_rmse': float(np.sqrt(mean_squared_error(y_gpa_train, train_gpa_pred))),
        'test_gpa_rmse': float(np.sqrt(mean_squared_error(y_gpa_test, test_gpa_pred)))
    }
    
    return model, latent_h_matrix.numpy(), predictions, diagnostics

if __name__ == "__main__":
    from data.generate_data import fetch_real_published_data
    static_df, tensor, _ = fetch_real_published_data()
    model, latent_h, preds, diagnostics = train_base_lstm(tensor, static_df, epochs=5)
    print(f"Bi-LSTM Base Model Trained Successfully.")
    print(f"Latent Hidden Matrix Shape: {latent_h.shape}")
    print(f"Sample Predicted Risk: {preds['pred_dropout_risk'][:5]}")
