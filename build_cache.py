"""
Pre-computation & Fast Cache Builder for CP-SSX Engine
Integrates Aligned PCA Factor Engine & PyTorch Multi-Task Bi-LSTM sequence encoder.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.generate_data import load_or_fetch_dataset
from src.models.base_lstm import train_base_lstm
from src.models.pca_factor_engine import PCAFactorEngine
from src.causal.uplift_engine import CausalUpliftEngine
from src.optimization.milp_allocator import MILPResourceAllocator

def precompute_and_cache(data_dir: str = "data_cache", force_rebuild: bool = False):
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "pipeline_cache.pkl")
    
    print("[CP-SSX Cache] Loading dataset...")
    static_df, clickstream_tensor, raw_concepts_df = load_or_fetch_dataset(data_dir=data_dir, force_rebuild=force_rebuild)
    
    print("[CP-SSX Cache] Fine-tuning PyTorch Multi-Task Bi-LSTM Sequence Model (15 epochs)...")
    base_model, latent_h_matrix, predictions, diagnostics = train_base_lstm(
        clickstream_tensor, static_df, epochs=15, batch_size=32
    )
    
    pred_risk = predictions["pred_dropout_risk"]
    pred_gpa = predictions["pred_final_gpa"]
    
    # Store predictions alongside empirical baseline risk
    static_df["pred_dropout_risk"] = np.round(pred_risk, 3)
    static_df["pred_final_gpa"] = np.round(pred_gpa, 2)
    
    print("[CP-SSX Cache] Fitting Aligned PCA Orthogonal Factor Decomposition Engine on latent states R^32...")
    pca_engine = PCAFactorEngine(n_components=4)
    pca_factors_df, pca_metrics = pca_engine.fit_transform(latent_h_matrix, reference_concepts_df=raw_concepts_df)
    
    probed_concepts_df = pd.DataFrame({
        "student_id": static_df["student_id"],
        "C1_comprehension_bottleneck": pca_factors_df["F1_academic_factor"],
        "C2_procrastination_accel": pca_factors_df["F2_procrastination_factor"],
        "C3_financial_hardship": pca_factors_df["F3_financial_factor"],
        "C4_peer_isolation": pca_factors_df["F4_social_factor"]
    })
    
    print("[CP-SSX Cache] Fitting Causal Uplift Estimator...")
    uplift_engine = CausalUpliftEngine()
    uplift_engine.fit(static_df, probed_concepts_df, static_df)
    cate_df = uplift_engine.predict_cate(static_df, probed_concepts_df)
    
    model_diagnostics = {
        **diagnostics,
        "pca_metrics": pca_metrics,
        "pca_engine": pca_engine
    }
    
    pipeline_data = {
        "static_df": static_df,
        "clickstream_tensor": clickstream_tensor,
        "probed_concepts_df": probed_concepts_df,
        "cate_df": cate_df,
        "model_diagnostics": model_diagnostics
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(pipeline_data, f)
        
    print(f"[CP-SSX Cache] Aligned PCA-derived pipeline saved to '{cache_path}'. Explained Var={pca_metrics['total_variance_explained']*100:.1f}%")

if __name__ == "__main__":
    precompute_and_cache(force_rebuild=True)
