"""
Module 5: Prescriptive Mixed-Integer Linear Programming (MILP) Allocator
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Solves optimal resource allocation across 4,424 students.
Guarantees physical bounds: Prescribed Uplift <= Base Risk - 0.02.
Prioritizes at-risk students (Base Risk >= 12%), preserving scarce resources for those in need.
"""

import numpy as np
import pandas as pd
import pulp
from typing import Dict, Any, Tuple

class MILPResourceAllocator:
    """
    Mixed-Integer Linear Program (MILP) Resource Allocator using PuLP & CBC Solver.
    """
    def __init__(self):
        pass

    def solve_allocation(
        self,
        static_df: pd.DataFrame,
        cate_df: pd.DataFrame,
        cap_advising_hours: float = 1200.0,
        cap_tutoring_hours: float = 2000.0,
        budget_grant_dollars: float = 150000.0,
        cost_advising_hrs: float = 2.0,
        cost_tutoring_hrs: float = 4.0,
        cost_grant_dollars: float = 500.0,
        min_risk_threshold: float = 0.12
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Solves MILP optimization problem for cohort resource allocation.
        """
        num_students = len(static_df)
        student_ids = static_df["student_id"].values
        base_risks = static_df["base_dropout_risk"].values
        
        treatments = ["None", "Advising", "Tutoring", "Micro-Grant"]
        
        ideal_interventions = []
        ideal_uplifts = []
        
        for i in range(num_students):
            br = float(base_risks[i])
            if br < min_risk_threshold:
                ideal_interventions.append("None (Low Risk / Safe)")
                ideal_uplifts.append(0.0)
            else:
                max_u = max(0.01, br - 0.02)
                adv_u = min(float(cate_df.iloc[i].get("CATE_Advising", 0.32)), max_u)
                tut_u = min(float(cate_df.iloc[i].get("CATE_Tutoring", 0.40)), max_u)
                grt_u = min(float(cate_df.iloc[i].get("CATE_Micro-Grant", 0.45)), max_u)
                
                u_map = {"Advising": adv_u, "Tutoring": tut_u, "Micro-Grant": grt_u}
                best_t = max(u_map, key=u_map.get)
                ideal_interventions.append(best_t)
                ideal_uplifts.append(u_map[best_t])

        # Define PuLP Integer Programming Model
        model = pulp.LpProblem("CP_SSX_Resource_Allocation", pulp.LpMaximize)
        
        x = {}
        for i in range(num_students):
            for t in treatments:
                x[i, t] = pulp.LpVariable(f"x_{i}_{t}", cat=pulp.LpBinary)
                
        objective_terms = []
        for i in range(num_students):
            br = float(base_risks[i])
            if br < min_risk_threshold:
                # Force low-risk safe students to None
                model += x[i, "None"] == 1, f"Force_None_Low_Risk_{i}"
                objective_terms.append(0.0 * x[i, "None"])
            else:
                max_u = max(0.01, br - 0.02)
                adv_u = min(float(cate_df.iloc[i].get("CATE_Advising", 0.32)), max_u)
                tut_u = min(float(cate_df.iloc[i].get("CATE_Tutoring", 0.40)), max_u)
                grt_u = min(float(cate_df.iloc[i].get("CATE_Micro-Grant", 0.45)), max_u)
                
                # Uplift coefficients (already bounded by base_risk - 0.02)
                objective_terms.append((adv_u * 100.0) * x[i, "Advising"])
                objective_terms.append((tut_u * 100.0) * x[i, "Tutoring"])
                objective_terms.append((grt_u * 100.0) * x[i, "Micro-Grant"])
                objective_terms.append(0.0 * x[i, "None"])
            
        model += pulp.lpSum(objective_terms), "Weighted_Risk_Reduced_Points"
        
        for i in range(num_students):
            model += pulp.lpSum([x[i, t] for t in treatments]) == 1, f"Single_Intervention_Student_{i}"
            
        model += (
            pulp.lpSum([x[i, "Advising"] * cost_advising_hrs for i in range(num_students)]) <= cap_advising_hours,
            "Advising_Capacity_Cap"
        )
        
        model += (
            pulp.lpSum([x[i, "Tutoring"] * cost_tutoring_hrs for i in range(num_students)]) <= cap_tutoring_hours,
            "Tutoring_Capacity_Cap"
        )
        
        model += (
            pulp.lpSum([x[i, "Micro-Grant"] * cost_grant_dollars for i in range(num_students)]) <= budget_grant_dollars,
            "Micro_Grant_Budget_Cap"
        )
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)
        
        if model.status != pulp.constants.LpStatusOptimal:
            print(f"[MILP Warning] Solver status: {pulp.LpStatus[model.status]}. Results may be suboptimal.")
        
        assigned_interventions = []
        assigned_uplifts = []
        is_capacity_constrained = []
        
        for i in range(num_students):
            br = float(base_risks[i])
            chosen = "None (Control / Unallocated)"
            chosen_uplift = 0.0
            
            for t in treatments:
                if pulp.value(x[i, t]) is not None and pulp.value(x[i, t]) > 0.5:
                    if t == "None":
                        chosen = "None (Control / Unallocated)"
                        chosen_uplift = 0.0
                    else:
                        chosen = t
                        max_u = max(0.01, br - 0.02)
                        if t == "Advising":
                            chosen_uplift = min(float(cate_df.iloc[i].get("CATE_Advising", 0.32)), max_u)
                        elif t == "Tutoring":
                            chosen_uplift = min(float(cate_df.iloc[i].get("CATE_Tutoring", 0.40)), max_u)
                        elif t == "Micro-Grant":
                            chosen_uplift = min(float(cate_df.iloc[i].get("CATE_Micro-Grant", 0.45)), max_u)
                    break
            
            assigned_interventions.append(chosen)
            assigned_uplifts.append(chosen_uplift)
            is_capacity_constrained.append(chosen != ideal_interventions[i])

        res_df = static_df.copy()
        res_df["assigned_intervention"] = assigned_interventions
        res_df["prescribed_uplift"] = np.round(assigned_uplifts, 3)
        res_df["ideal_intervention"] = ideal_interventions
        res_df["ideal_uplift"] = np.round(ideal_uplifts, 3)
        res_df["is_capacity_constrained"] = is_capacity_constrained
        res_df["post_intervention_risk"] = np.round(np.maximum(0.02, res_df["base_dropout_risk"] - res_df["prescribed_uplift"]), 3)

        adv_count = sum(1 for t in assigned_interventions if t == "Advising")
        tut_count = sum(1 for t in assigned_interventions if t == "Tutoring")
        grant_count = sum(1 for t in assigned_interventions if t == "Micro-Grant")
        
        adv_hours = adv_count * cost_advising_hrs
        tut_hours = tut_count * cost_tutoring_hrs
        grant_dollars = grant_count * cost_grant_dollars
        
        allocated_stus = adv_count + tut_count + grant_count
        total_risk_pts = float(np.sum(assigned_uplifts) * 100.0)
        avg_uplift_pct = (total_risk_pts / max(1, allocated_stus)) if allocated_stus > 0 else 0.0

        summary = {
            "status": pulp.LpStatus[model.status],
            "total_students": num_students,
            "allocated_students": allocated_stus,
            "total_risk_reduced_points": total_risk_pts,
            "avg_uplift_per_student_pct": avg_uplift_pct,
            "advising_count": adv_count,
            "advising_hours_used": adv_hours,
            "advising_hours_cap": cap_advising_hours,
            "advising_utilization_pct": (adv_hours / cap_advising_hours) * 100.0,
            "tutoring_count": tut_count,
            "tutoring_hours_used": tut_hours,
            "tutoring_hours_cap": cap_tutoring_hours,
            "tutoring_utilization_pct": (tut_hours / cap_tutoring_hours) * 100.0,
            "grant_count": grant_count,
            "grant_dollars_used": grant_dollars,
            "grant_dollars_cap": budget_grant_dollars,
            "grant_utilization_pct": (grant_dollars / budget_grant_dollars) * 100.0
        }
        
        return res_df, summary
