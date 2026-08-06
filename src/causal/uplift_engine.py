"""
Module 4: Causal Uplift (CATE) & Counterfactual Recourse Engine
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Estimates heterogeneous treatment effects (tau_{i,a}) for multi-treatment interventions
a in {Advising, Tutoring, Micro-Grant} using T-Learners (Gradient Boosting / Random Forest).
Generates actionable, minimal behavioral counterfactual recourse targets for students.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional

class CausalUpliftEngine:
    """
    Multi-Treatment Conditional Average Treatment Effect (CATE) Estimator
    using T-Learner Meta-Algorithms with Gradient Boosting base regressors.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.treatments = ["Advising", "Tutoring", "Micro-Grant"]
        
        # Base estimators for control and treatments
        self.models_control: Dict[str, GradientBoostingRegressor] = {}
        self.models_treated: Dict[str, GradientBoostingRegressor] = {}
        self.feature_names: List[str] = []

    def fit(
        self,
        X_df: pd.DataFrame,
        concepts_df: pd.DataFrame,
        static_df: pd.DataFrame
    ) -> "CausalUpliftEngine":
        """
        Fits T-Learners to estimate uplift tau_{i,a} for each intervention type.
        
        Parameters:
            X_df (pd.DataFrame): Input student feature matrix.
            concepts_df (pd.DataFrame): Probed concept features (C1...C4).
            static_df (pd.DataFrame): Ground truth static records with uplift outcomes.
        """
        # Combine static features and concepts into feature set
        feature_cols = [
            "prior_gpa", "poverty_index", "study_hours_budget",
            "mean_engagement_velocity", "mean_procrastination_lag", "academic_preparedness_index"
        ]
        concept_cols = [
            "C1_comprehension_bottleneck", "C2_procrastination_accel",
            "C3_financial_hardship", "C4_peer_isolation"
        ]

        
        full_X = pd.concat([
            X_df[feature_cols].reset_index(drop=True),
            concepts_df[concept_cols].reset_index(drop=True)
        ], axis=1)
        
        self.feature_names = list(full_X.columns)
        X_scaled = self.scaler.fit_transform(full_X)
        
        # Fit T-Learners for each treatment using synthetic uplift targets
        for tr in self.treatments:
            target_col = f"tau_{tr.lower().replace('-', '_')}"
            if target_col in static_df.columns:
                tau_target = static_df[target_col].values
            elif tr == "Micro-Grant" and "tau_grant" in static_df.columns:
                tau_target = static_df["tau_grant"].values
            else:
                # Synthetic proxy calculation if column missing
                tau_target = 0.25 * concepts_df["C1_comprehension_bottleneck"].values
                
            Y_control = static_df['base_dropout_risk'].values
            Y_treated = np.maximum(0.02, static_df['base_dropout_risk'].values - tau_target)
            
            model_control = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, random_state=self.random_state)
            model_control.fit(X_scaled, Y_control)
            self.models_control[tr] = model_control
            
            model_treated = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, random_state=self.random_state)
            model_treated.fit(X_scaled, Y_treated)
            self.models_treated[tr] = model_treated
            
        return self

    def predict_cate(
        self,
        X_df: pd.DataFrame,
        concepts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predicts treatment effect tau_{i,a} for all interventions.
        
        Returns:
            pd.DataFrame: DataFrame containing predicted CATE values for Advising, Tutoring, and Micro-Grant.
        """
        feature_cols = [
            "prior_gpa", "poverty_index", "study_hours_budget",
            "mean_engagement_velocity", "mean_procrastination_lag", "academic_preparedness_index"
        ]
        concept_cols = [
            "C1_comprehension_bottleneck", "C2_procrastination_accel",
            "C3_financial_hardship", "C4_peer_isolation"
        ]

        
        full_X = pd.concat([
            X_df[feature_cols].reset_index(drop=True),
            concepts_df[concept_cols].reset_index(drop=True)
        ], axis=1)
        
        X_scaled = self.scaler.transform(full_X)
        cate_preds = {}
        
        for tr in self.treatments:
            tau_hat = self.models_control[tr].predict(X_scaled) - self.models_treated[tr].predict(X_scaled)
            # Clip uplift between 0.00 and 0.65 for realistic intervention impact
            cate_preds[f"CATE_{tr}"] = np.clip(tau_hat, 0.01, 0.65)
            
        return pd.DataFrame(cate_preds)

class CounterfactualRecourseEngine:
    """
    Generates actionable, non-stigmatizing counterfactual recourse targets for students
    seeking to lower their risk score below a safety threshold.
    """
    def __init__(self, target_threshold: float = 0.20):
        self.target_threshold = target_threshold

    def generate_recourse(
        self,
        student_row: pd.Series,
        concept_row: pd.Series,
        current_risk: float
    ) -> Dict[str, Any]:
        """
        Computes minimal actionable behavioral shifts to reach safe risk status.
        
        Parameters:
            student_row (pd.Series): Static feature values for the student.
            concept_row (pd.Series): Probed concept values for the student.
            current_risk (float): Current dropout risk prediction.
            
        Returns:
            Dict[str, Any]: Structured recourse recommendations.
        """
        if current_risk <= self.target_threshold:
            return {
                "status": "Safe",
                "message": "Student is currently below the dropout risk threshold. Maintain current academic performance.",
                "actions": []
            }
            
        risk_gap = current_risk - self.target_threshold
        actions = []
        
        # Analyze primary driving concept bottleneck
        c1 = concept_row.get("C1_comprehension_bottleneck", 0.0)
        c2 = concept_row.get("C2_procrastination_accel", 0.0)
        c3 = concept_row.get("C3_financial_hardship", 0.0)
        c4 = concept_row.get("C4_peer_isolation", 0.0)
        
        # Recourse Rule 1: Procrastination & Engagement Velocity
        if c2 > 0.4:
            current_lag = student_row.get("mean_procrastination_lag", 2.0)
            target_lag = max(0.0, np.round(current_lag * 0.4, 1))
            impact_val = c2 * 0.4
            actions.append({
                "metric": "Submission Delay",
                "current": f"{current_lag:.1f} days past deadline",
                "target": f"{target_lag:.1f} days (Submit assignments prior to deadline)",
                "impact": f"-{impact_val:.2f} risk points"
            })
            
        # Recourse Rule 2: Study Hours & Peer Engagement
        if c4 > 0.4 or c1 > 0.4:
            current_hours = student_row.get("study_hours_budget", 15)
            target_hours = int(current_hours + 5)
            impact_val = max(c1, c4) * 0.3
            actions.append({
                "metric": "Weekly Peer & Study Hours",
                "current": f"{current_hours} hrs/week",
                "target": f"{target_hours} hrs/week (Join weekly peer study circle)",
                "impact": f"-{impact_val:.2f} risk points"
            })
            
        # Recourse Rule 3: Financial Aid Support
        if c3 > 0.4:
            impact_val = c3 * 0.5
            actions.append({
                "metric": "Financial Hardship Relief",
                "current": "High cost burden / low study budget",
                "target": "Apply for Campus Micro-Grant & Work-Study stipend",
                "impact": f"-{impact_val:.2f} risk points"
            })

            
        if not actions:
            actions.append({
                "metric": "Academic Advising",
                "current": "Baseline engagement",
                "target": "Schedule bi-weekly 1-on-1 coaching session with academic advisor",
                "impact": f"-{current_risk * 0.2:.2f} risk points"
            })
            
        return {
            "status": "Intervention Recommended",
            "current_risk": float(current_risk),
            "target_threshold": self.target_threshold,
            "risk_gap": float(np.round(risk_gap, 3)),
            "actions": actions
        }

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.generate_data import load_or_fetch_dataset
    static_df, tensor, concepts_df = load_or_fetch_dataset()
    
    uplift_engine = CausalUpliftEngine()
    uplift_engine.fit(static_df, concepts_df, static_df)
    cate_df = uplift_engine.predict_cate(static_df, concepts_df)
    print("CATE Uplift Predictions Sample:\n", cate_df.head())
    
    recourse_engine = CounterfactualRecourseEngine(target_threshold=0.20)
    recourse = recourse_engine.generate_recourse(static_df.iloc[0], concepts_df.iloc[0], current_risk=0.45)
    print("\nCounterfactual Recourse Sample:\n", recourse)

