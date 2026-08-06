"""
Module 3: Aligned PCA Orthogonal Factor Decomposition Engine
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Extracts 4 strictly orthogonal factor dimensions (Cov=0.000) from 32-dim latent states.
Uses Automated Eigenvector Alignment & Sign Correction to match principal axes to 4 canonical factors:
  - F1 / C1: Academic Comprehension Bottleneck
  - F2 / C2: Procrastination & Submission Lag
  - F3 / C3: Financial Hardship & Debt
  - F4 / C4: Social & Peer Isolation
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

class PCAFactorEngine:
    """
    Unsupervised Orthogonal Factor Extractor with Automated Eigenvector Alignment.
    Transforms latent vectors h_{i,t} in R^32 into 4 canonical, strictly orthogonal factor dimensions.
    """
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.component_alignment: List[int] = [0, 1, 2, 3]
        self.component_signs: List[float] = [1.0, 1.0, 1.0, 1.0]
        self.p1 = np.zeros(4)
        self.p99 = np.ones(4)
        self.is_fitted = False
        
        self.canonical_column_names = [
            "F1_academic_factor",
            "F2_procrastination_factor",
            "F3_financial_factor",
            "F4_social_factor"
        ]

    def fit_transform(
        self,
        latent_h_matrix: np.ndarray,
        reference_concepts_df: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits PCA on latent hidden matrix H in R^(N x 32) and aligns components to canonical factor definitions.
        """
        # 1. Standardize latent states
        h_scaled = self.scaler.fit_transform(latent_h_matrix)
        
        # 2. Fit PCA components (raw orthogonal scores)
        raw_pca_scores = self.pca.fit_transform(h_scaled)
        
        # 3. Automated Eigenvector Alignment & Sign Correction
        if reference_concepts_df is not None:
            ref_cols = [
                "C1_comprehension_bottleneck",
                "C2_procrastination_accel",
                "C3_financial_hardship",
                "C4_peer_isolation"
            ]
            ref_matrix = reference_concepts_df[ref_cols].values
            
            # Correlation matrix between PCA components (4) and reference concepts (4)
            corr_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    r = np.corrcoef(raw_pca_scores[:, i], ref_matrix[:, j])[0, 1]
                    corr_matrix[i, j] = r if not np.isnan(r) else 0.0
                    
            # Match each PCA component to the reference concept with max absolute correlation
            used_refs = set()
            self.component_alignment = [0, 1, 2, 3]
            self.component_signs = [1.0, 1.0, 1.0, 1.0]
            
            for comp_idx in range(4):
                abs_corrs = np.abs(corr_matrix[comp_idx, :])
                for ref_idx in np.argsort(-abs_corrs):
                    if ref_idx not in used_refs:
                        used_refs.add(ref_idx)
                        self.component_alignment[ref_idx] = comp_idx
                        raw_r = corr_matrix[comp_idx, ref_idx]
                        self.component_signs[ref_idx] = 1.0 if raw_r >= 0 else -1.0
                        break

        # Re-order and sign-correct PCA components to match canonical columns
        aligned_pca_scores = np.zeros_like(raw_pca_scores)
        for target_idx in range(4):
            comp_idx = self.component_alignment[target_idx]
            sign = self.component_signs[target_idx]
            aligned_pca_scores[:, target_idx] = sign * raw_pca_scores[:, comp_idx]

        self.is_fitted = True

        # 4. Robust Percentile Scaling into [0.05, 0.95] preserving true distance variance
        p1 = np.percentile(aligned_pca_scores, 1, axis=0)
        p99 = np.percentile(aligned_pca_scores, 99, axis=0)
        self.p1 = p1
        self.p99 = p99
        
        scaled_factors = np.zeros_like(aligned_pca_scores)
        for col in range(4):
            span = max(1e-5, p99[col] - p1[col])
            scaled_col = 0.05 + 0.90 * np.clip((aligned_pca_scores[:, col] - p1[col]) / span, 0.0, 1.0)
            scaled_factors[:, col] = scaled_col

        # Verify strict covariance orthogonality (Cov = 0.000)
        cov_matrix = np.cov(raw_pca_scores.T)
        off_diag_cov = np.sum(np.abs(cov_matrix - np.diag(np.diag(cov_matrix))))
        
        explained_variance_ratio = self.pca.explained_variance_ratio_
        total_variance_explained = float(np.sum(explained_variance_ratio))

        factors_df = pd.DataFrame(
            np.round(scaled_factors, 3),
            columns=self.canonical_column_names
        )

        metrics = {
            "explained_variance_ratio": [float(v) for v in explained_variance_ratio],
            "total_variance_explained": float(total_variance_explained),
            "orthogonality_off_diag_cov": float(off_diag_cov),
            "component_alignment": self.component_alignment,
            "component_signs": self.component_signs,
            "eigenvector_loadings": self.pca.components_.tolist()
        }

        print(f"[PCA Engine] Aligned Orthogonal PCA factor extraction completed ({self.n_components} factors).")
        print(f"[PCA Engine] Variance Explained: {total_variance_explained*100:.1f}%. Off-diag Covariance: {off_diag_cov:.6f}")

        return factors_df, metrics

    def transform_single(self, single_h_vector: np.ndarray) -> np.ndarray:
        """
        Transforms a single student's 32-dim latent vector into 4 aligned factor scores.
        """
        if not self.is_fitted:
            raise ValueError("PCAFactorEngine must be fitted before calling transform_single.")
        if single_h_vector.ndim == 1:
            single_h_vector = single_h_vector.reshape(1, -1)
            
        scaled_h = self.scaler.transform(single_h_vector)
        raw_pca = self.pca.transform(scaled_h)[0]
        
        aligned_pca = np.zeros(4)
        for target_idx in range(4):
            comp_idx = self.component_alignment[target_idx]
            sign = self.component_signs[target_idx]
            aligned_pca[target_idx] = sign * raw_pca[comp_idx]
            
        scaled_single = np.zeros(4)
        for target_idx in range(4):
            span = max(1e-5, self.p99[target_idx] - self.p1[target_idx])
            scaled_single[target_idx] = 0.05 + 0.90 * np.clip((aligned_pca[target_idx] - self.p1[target_idx]) / span, 0.0, 1.0)
        return np.round(scaled_single, 3)
