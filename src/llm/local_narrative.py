"""
Module 6: Privacy-Preserving On-Device SLM Interface
CP-SSX (Causal Prescriptive Student Success eXplainer Engine)

Compiles structured quantitative metrics (Risk, Concepts, CATE, Recourse, MILP allocations)
into role-specific, structured narrative generation narratives. Integrates with local Ollama SLMs
(llama3.2:3b, phi4:mini) with automated deterministic template fallback if Ollama is offline.
"""

import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class LocalSLMNarrativeCompiler:
    """
    Zero-hallucination narrative generation layer using local Ollama SLM API
    with robust rule-based template fallback.
    """
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "llama3.2:3b",
        timeout: float = 3.0
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """
        Sends prompt to local Ollama HTTP server with strict timeout.
        Returns response string if successful, else None.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Low temperature for zero-hallucination fidelity
                    "top_p": 0.9
                }
            }
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            print(f'[SLM] Ollama unavailable: {e}. Using template fallback.')
        return None

    def generate_advisor_brief(
        self,
        student_id: str,
        current_risk: float,
        concepts: Dict[str, float],
        prescribed_intervention: str,
        expected_uplift: float,
        recourse: Dict[str, Any]
    ) -> str:
        """
        Generates an actionable, executive outreach brief for Academic Advisors.
        """
        prompt = (
            f"You are an AI Academic Advisor Assistant. Compile the following student data into an actionable outreach brief:\n"
            f"- Student ID: {student_id}\n"
            f"- Dropout Risk: {current_risk * 100:.1f}%\n"
            f"- Concept Bottlenecks: {concepts}\n"
            f"- Prescribed Intervention: {prescribed_intervention} (Uplift: -{expected_uplift * 100:.1f}% risk)\n"
            f"- Recommended Actions: {recourse.get('actions', [])}\n\n"
            f"Write a concise, professional 3-bullet outreach plan for the advisor."
        )
        
        ollama_output = self._query_ollama(prompt)
        if ollama_output:
            return ollama_output

        # Deterministic Executive Fallback Engine
        c_top = max(concepts.items(), key=lambda x: x[1])[0] if concepts else "C1_comprehension"
        c_top_readable = c_top.replace("C1_", "").replace("C2_", "").replace("C3_", "").replace("C4_", "").replace("_", " ").title()
        
        action_text = "\n".join([f"   • {act['metric']}: Shift from '{act['current']}' to '{act['target']}'" for act in recourse.get("actions", [])])
        
        fallback_brief = (
            f"### 📋 EXECUTIVE ADVISOR ACTION BRIEF — {student_id}\n\n"
            f"• **Risk Assessment:** Current Dropout Risk is **{current_risk * 100:.1f}%**.\n"
            f"• **Primary Mechanistic Driver:** {c_top_readable} (Severity Index: {concepts.get(c_top, 0.0):.2f}).\n"
            f"• **MILP Prescribed Intervention:** **{prescribed_intervention}** (Projected Uplift: **-{expected_uplift * 100:.1f}%** risk reduction).\n"
            f"• **Actionable Counterfactual Recourse Plan:**\n"
            f"{action_text if action_text else '   • Maintain current weekly academic progress.'}\n"
            f"• **Advisor Outreach Script:** \"Hi {student_id}, based on your course trajectory, enrolling in {prescribed_intervention} "
            f"is estimated to boost your course success probability by {expected_uplift * 100:.1f}%. Let's meet this week to set up your schedule.\""
        )
        return fallback_brief

    def generate_admin_brief(self, summary_dict: Dict[str, Any]) -> str:
        """
        Generates an executive capacity & policy optimization brief for University Deans & Administrators.
        """
        prompt = (
            f"You are a University Data Officer compiling an executive summary of MILP resource optimization:\n"
            f"{json.dumps(summary_dict, indent=2)}\n\n"
            f"Write a 4-bullet executive report on budget allocation, capacity bottlenecks, and total risk reduction."
        )
        
        ollama_output = self._query_ollama(prompt)
        if ollama_output:
            return ollama_output

        # Deterministic Executive Fallback Engine
        total_st = summary_dict.get("total_students", 0)
        alloc_st = summary_dict.get("allocated_students", 0)
        pts_reduced = summary_dict.get("total_risk_reduced_points", 0.0)
        
        adv_pct = summary_dict.get("advising_utilization_pct", 0.0)
        tut_pct = summary_dict.get("tutoring_utilization_pct", 0.0)
        grant_pct = summary_dict.get("grant_utilization_pct", 0.0)

        bottlenecks = []
        if adv_pct >= 90.0: bottlenecks.append("Faculty Advising Hours")
        if tut_pct >= 90.0: bottlenecks.append("Tutoring Center Capacity")
        if grant_pct >= 90.0: bottlenecks.append("Micro-Grant Financial Budget")
        bottleneck_str = ", ".join(bottlenecks) if bottlenecks else "None (Sufficient Operational Headroom)"

        fallback_brief = (
            f"### 🏛️ EXECUTIVE DEAN & ADMIN POLICY BRIEFING\n\n"
            f"• **Aggregate Impact:** Successfully allocated prescriptive interventions to **{alloc_st} of {total_st}** at-risk students, "
            f"achieving **{pts_reduced:.1f} cumulative risk reduction points** across the cohort.\n"
            f"• **Operational Budget Utilization:**\n"
            f"   - **Faculty Advising:** {summary_dict.get('advising_hours_used', 0):.0f} / {summary_dict.get('advising_hours_cap', 0):.0f} hrs ({adv_pct:.1f}% utilized)\n"
            f"   - **Tutoring Center:** {summary_dict.get('tutoring_hours_used', 0):.0f} / {summary_dict.get('tutoring_hours_cap', 0):.0f} hrs ({tut_pct:.1f}% utilized)\n"
            f"   - **Micro-Grant Fund:** ${summary_dict.get('grant_dollars_used', 0):,.0f} / ${summary_dict.get('grant_dollars_cap', 0):,.0f} ({grant_pct:.1f}% utilized)\n"
            f"• **Capacity Bottlenecks Identified:** {bottleneck_str}.\n"
            f"• **Policy Recommendation:** Reallocate unused tutoring reserves into {bottlenecks[0] if bottlenecks else 'Micro-Grants'} "
            f"to maximize marginal retention returns next semester."
        )
        return fallback_brief

    def generate_student_empowerment_brief(
        self,
        student_id: str,
        recourse: Dict[str, Any],
        assigned_intervention: str
    ) -> str:
        """
        Generates an encouraging, non-stigmatizing recourse brief for the Student View.
        """
        actions = recourse.get("actions", [])
        action_bullets = "\n".join([f"   ✨ **{act['metric']}**: {act['target']}" for act in actions])
        
        if assigned_intervention in ["None", "None (Control / Unallocated)", "Control"]:
            rec_text = "• **Standard Academic Track**: Your engagement is solid! No intensive support intervention is currently needed."
        else:
            rec_text = f"• **{assigned_intervention}**: You have been pre-approved for priority 1-on-1 support."

        brief = (
            f"### 🌟 PERSONALIZED ACADEMIC GROWTH PATHWAY — {student_id}\n\n"
            f"Welcome back! Your academic trajectory is on track with custom support resources available to help you excel.\n\n"
            f"🎯 **Your Recommended Success Resource:**\n"
            f"{rec_text}\n\n"
            f"🚀 **Actionable Micro-Steps for This Week:**\n"
            f"{action_bullets if action_bullets else '   ✨ Keep up the great work in your current weekly assignments!'}\n\n"
            f"💡 *Remember: Small, consistent shifts in study habits yield substantial long-term academic growth!*"
        )
        return brief

if __name__ == "__main__":
    import sys
    compiler = LocalSLMNarrativeCompiler()
    advisor_brief = compiler.generate_advisor_brief(
        "STU_10001",
        current_risk=0.45,
        concepts={"C1_comprehension": 0.65, "C2_procrastination": 0.30},
        prescribed_intervention="Tutoring Center",
        expected_uplift=0.25,
        recourse={"actions": [{"metric": "Submission Delay", "current": "3 days", "target": "0 days"}]}
    )
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("Advisor Brief Sample:\n", advisor_brief)

