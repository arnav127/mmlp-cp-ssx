"""
Module 3 (Supervised Alternative): Mechanistic Concept-Bottleneck Explainer (MC-PRE)
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Implements a Supervised Concept Probing Layer (PyTorch) as a neural baseline alternative to
the Unsupervised PCA Factor Engine. Attaches to hidden bottleneck states h_{i,t} in R^32 
to decode managerial concepts C_k in [0, 1]^4:
  - C1: Concept Comprehension Bottleneck
  - C2: Procrastination Acceleration
  - C3: Social / Peer Isolation
  - C4: Financial Hardship Proxy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class ConceptProbingLayer(nn.Module):
    """
    Mechanistic Concept Probing Layer (MC-PRE).
    Maps high-dimensional latent hidden states h_{i,t} in R^32 onto 4 interpretable concept logits.
    """
    def __init__(self, bottleneck_dim: int = 32, num_concepts: int = 4):
        super(ConceptProbingLayer, self).__init__()
        
        self.probing_net = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, num_concepts),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting concept scores C_1...C_4 in [0, 1].
        
        Parameters:
            h (torch.Tensor): Latent representation matrix (batch_size, 32)
            
        Returns:
            torch.Tensor: Concept prediction matrix (batch_size, 4)
        """
        return self.probing_net(h)

def train_concept_prober(
    latent_h_matrix: np.ndarray,
    concepts_df: pd.DataFrame,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.005
) -> Tuple[ConceptProbingLayer, np.ndarray, Dict[str, float]]:
    """
    Trains the concept probing layer on frozen bottleneck latent representations.
    
    Parameters:
        latent_h_matrix (np.ndarray): Extracted latent representations matrix H (N, 32).
        concepts_df (pd.DataFrame): Ground truth managerial concept labels.
        
    Returns:
        Tuple[ConceptProbingLayer, np.ndarray, Dict[str, float]]:
            - Trained ConceptProbingLayer model
            - Predicted concept matrix C_pred (N, 4)
            - Concept probe accuracy/correlation metrics dictionary
    """
    concept_cols = [
        "C1_comprehension_bottleneck",
        "C2_procrastination_accel",
        "C3_financial_hardship",
        "C4_peer_isolation"
    ]
    
    H_tensor = torch.tensor(latent_h_matrix, dtype=torch.float32)
    C_tensor = torch.tensor(concepts_df[concept_cols].values, dtype=torch.float32)
    
    dataset = TensorDataset(H_tensor, C_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    prober = ConceptProbingLayer(bottleneck_dim=latent_h_matrix.shape[1], num_concepts=4)
    optimizer = optim.Adam(prober.parameters(), lr=lr)
    bce_criterion = nn.BCELoss()
    
    prober.train()
    for epoch in range(epochs):
        for b_h, b_c in dataloader:
            optimizer.zero_grad()
            c_pred = prober(b_h)
            loss = bce_criterion(c_pred, b_c)
            loss.backward()
            optimizer.step()
            
    prober.eval()
    with torch.no_grad():
        c_pred_all = prober(H_tensor).numpy()
        
    # Calculate concept evaluation correlations (Pearson r per concept)
    metrics = {}
    for idx, col_name in enumerate(concept_cols):
        true_vals = concepts_df[col_name].values
        pred_vals = c_pred_all[:, idx]
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        metrics[f"r_{col_name}"] = float(np.round(corr, 3))
        
    return prober, c_pred_all, metrics

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.generate_data import load_or_fetch_dataset
    from src.models.base_lstm import train_base_lstm
    
    static_df, clickstream_tensor, concepts_df = load_or_fetch_dataset()
    _, latent_h, _ = train_base_lstm(clickstream_tensor, static_df, epochs=3)
    prober, c_preds, metrics = train_concept_prober(latent_h, concepts_df, epochs=5)
    print("Concept Prober Trained Successfully.")
    print("Concept Correlations with Ground Truth:", metrics)
    print("Predicted Concepts Sample:\n", c_preds[:3])

