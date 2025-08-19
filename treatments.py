import streamlit as st
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from baysian import BayesianNetworkBuilder


class BNTreatmentInference:
    """
    Uses existing Bayesian Network to suggest treatments through backwards reasoning.
    Identifies which symptoms/factors, if modified, would most reduce disease probability.
    """

    def __init__(self, bn_builder: BayesianNetworkBuilder):
        self.bn = bn_builder
        self.model = bn_builder.model

    def analyze_treatment_targets(self, target_condition: str, current_evidence: Dict[str, any]) -> Dict[str, any]:
        """
        Analyze which interventions would be most effective by:
        1. Finding current disease probability
        2. Testing what happens if we "treat" (flip) each positive symptom
        3. Ranking interventions by impact
        """

        # Get baseline probability with current symptoms
        baseline_probs, _ = self.bn.infer_with_evidence_filtering(target_condition, current_evidence)
        baseline_risk = baseline_probs.get("present", 0.0)

        if baseline_risk < 0.1:
            return {"message": "Low risk detected - focus on prevention", "interventions": []}

        # Find all positive symptoms (present = 1 or "present")
        positive_symptoms = []
        for symptom, value in current_evidence.items():
            if self._is_positive_evidence(value):
                positive_symptoms.append(symptom)

        if not positive_symptoms:
            return {"message": "No active symptoms to address", "interventions": []}

        # Test impact of "treating" each symptom
        interventions = []

        for symptom in positive_symptoms:
            impact = self._calculate_intervention_impact(
                target_condition, current_evidence, symptom, baseline_risk
            )
            if impact:
                interventions.append(impact)

        # Sort by effectiveness (highest risk reduction first)
        interventions.sort(key=lambda x: x["risk_reduction"], reverse=True)

        return {
            "baseline_risk": baseline_risk,
            "interventions": interventions[:5],  # Top 5 most effective
            "message": f"Found {len(interventions)} potential intervention targets"
        }

    def _is_positive_evidence(self, value) -> bool:
        """Check if evidence represents a positive/present symptom."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value >= 0.5
        if isinstance(value, str):
            return value.lower().strip() in ['present', 'yes', 'true', '1']
        return False

    def _calculate_intervention_impact(self, target_condition: str, current_evidence: Dict[str, any],
                                       symptom_to_treat: str, baseline_risk: float) -> Optional[Dict[str, any]]:
        """Calculate the impact of 'treating' (removing) a specific symptom."""

        try:
            # Create hypothetical evidence where this symptom is "treated" (set to absent)
            treated_evidence = current_evidence.copy()
            treated_evidence[symptom_to_treat] = "absent"  # or 0

            # Calculate new risk with this symptom "treated"
            new_probs, info = self.bn.infer_with_evidence_filtering(target_condition, treated_evidence)
            new_risk = new_probs.get("present", 0.0)

            # Calculate risk reduction
            risk_reduction = baseline_risk - new_risk
            relative_reduction = (risk_reduction / baseline_risk) * 100 if baseline_risk > 0 else 0

            # Skip if no meaningful impact
            if risk_reduction < 0.01:  # Less than 1% absolute reduction
                return None

            return {
                "symptom": symptom_to_treat,
                "baseline_risk": baseline_risk,
                "treated_risk": new_risk,
                "risk_reduction": risk_reduction,
                "relative_reduction": relative_reduction,
                "priority": self._get_priority_level(relative_reduction),
                "intervention_category": self._categorize_intervention(symptom_to_treat)
            }

        except Exception as e:
            st.warning(f"Could not analyze intervention for {symptom_to_treat}: {e}")
            return None

    def _get_priority_level(self, relative_reduction: float) -> str:
        """Assign priority level based on relative risk reduction."""
        if relative_reduction >= 30:
            return "High"
        elif relative_reduction >= 15:
            return "Medium"
        elif relative_reduction >= 5:
            return "Low"
        else:
            return "Minimal"

    def _categorize_intervention(self, symptom: str) -> str:
        """Categorize the type of intervention based on symptom name."""
        symptom_lower = symptom.lower()

        # Lifestyle interventions
        if any(term in symptom_lower for term in ['weight', 'exercise', 'diet', 'smoking', 'alcohol']):
            return "Lifestyle"

        # Medication targets
        if any(term in symptom_lower for term in ['pain', 'pressure', 'inflammation', 'infection']):
            return "Medication"

        # Physical therapy
        if any(term in symptom_lower for term in ['mobility', 'strength', 'balance', 'stiffness']):
            return "Physical Therapy"

        # Surgical consideration
        if any(term in symptom_lower for term in ['blockage', 'structural', 'mechanical']):
            return "Surgical"

        return "General Medical"

    def find_symptom_interactions(self, target_condition: str, current_evidence: Dict[str, any]) -> List[
        Dict[str, any]]:
        """
        Find combinations of symptoms that, if treated together, would have synergistic effects.
        """
        positive_symptoms = [s for s, v in current_evidence.items() if self._is_positive_evidence(v)]

        if len(positive_symptoms) < 2:
            return []

        combinations = []

        # Test pairs of symptoms
        for i, symptom1 in enumerate(positive_symptoms):
            for symptom2 in positive_symptoms[i + 1:]:
                combo_impact = self._test_combination_impact(
                    target_condition, current_evidence, [symptom1, symptom2]
                )
                if combo_impact:
                    combinations.append(combo_impact)

        return sorted(combinations, key=lambda x: x["combined_reduction"], reverse=True)[:3]

    def _test_combination_impact(self, target_condition: str, current_evidence: Dict[str, any],
                                 symptoms_to_treat: List[str]) -> Optional[Dict[str, any]]:
        """Test the combined impact of treating multiple symptoms together."""

        try:
            # Get baseline
            baseline_probs, _ = self.bn.infer_with_evidence_filtering(target_condition, current_evidence)
            baseline_risk = baseline_probs.get("present", 0.0)

            # Create evidence with all symptoms treated
            treated_evidence = current_evidence.copy()
            for symptom in symptoms_to_treat:
                treated_evidence[symptom] = "absent"

            new_probs, _ = self.bn.infer_with_evidence_filtering(target_condition, treated_evidence)
            new_risk = new_probs.get("present", 0.0)

            combined_reduction = baseline_risk - new_risk

            if combined_reduction < 0.05:  # Less than 5% reduction
                return None

            return {
                "symptoms": symptoms_to_treat,
                "combined_reduction": combined_reduction,
                "relative_reduction": (combined_reduction / baseline_risk) * 100 if baseline_risk > 0 else 0,
                "synergy": self._calculate_synergy(current_evidence, target_condition, symptoms_to_treat)
            }

        except Exception:
            return None

    def _calculate_synergy(self, current_evidence: Dict[str, any], target_condition: str,
                           symptoms: List[str]) -> float:
        """Calculate if treating symptoms together is more effective than individually."""

        try:
            # Get individual impacts
            individual_total = 0
            for symptom in symptoms:
                impact = self._calculate_intervention_impact(target_condition, current_evidence, symptom, 0.5)
                if impact:
                    individual_total += impact["risk_reduction"]

            # Get combined impact
            combo_impact = self._test_combination_impact(target_condition, current_evidence, symptoms)
            combined_total = combo_impact["combined_reduction"] if combo_impact else 0

            # Synergy = combined effect - sum of individual effects
            synergy = combined_total - individual_total
            return max(0, synergy)  # Only positive synergy

        except Exception:
            return 0

    def generate_treatment_suggestions(self, condition: str, evidence: Dict[str, any]) -> Dict[str, any]:
        """
        Main method to generate treatment suggestions using BN reasoning.
        """

        # Analyze individual interventions
        analysis = self.analyze_treatment_targets(condition, evidence)

        # Find combination effects
        combinations = self.find_symptom_interactions(condition, evidence)

        # Generate prioritized suggestions
        suggestions = self._format_treatment_suggestions(analysis, combinations)

        return {
            "condition": condition,
            "baseline_risk": analysis.get("baseline_risk", 0),
            "individual_interventions": analysis.get("interventions", []),
            "combination_interventions": combinations,
            "treatment_suggestions": suggestions,
            "disclaimer": self._get_medical_disclaimer()
        }

    def _format_treatment_suggestions(self, analysis: Dict[str, any],
                                      combinations: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Format the analysis into actionable treatment suggestions."""

        suggestions = []

        # High-priority individual interventions
        interventions = analysis.get("interventions", [])
        for intervention in interventions[:3]:  # Top 3
            if intervention["priority"] in ["High", "Medium"]:
                suggestions.append({
                    "type": "individual",
                    "priority": intervention["priority"],
                    "target": intervention["symptom"],
                    "category": intervention["intervention_category"],
                    "expected_benefit": f"{intervention['relative_reduction']:.1f}% risk reduction",
                    "description": self._generate_intervention_description(intervention)
                })

        # Combination interventions
        for combo in combinations[:2]:  # Top 2 combinations
            if combo["relative_reduction"] > 20:  # Meaningful reduction
                suggestions.append({
                    "type": "combination",
                    "priority": "High" if combo["relative_reduction"] > 40 else "Medium",
                    "targets": combo["symptoms"],
                    "expected_benefit": f"{combo['relative_reduction']:.1f}% risk reduction",
                    "synergy": combo["synergy"] > 0.02,  # 2% synergistic benefit
                    "description": f"Addressing {' and '.join(combo['symptoms'])} together"
                })

        return suggestions

    def _generate_intervention_description(self, intervention: Dict[str, any]) -> str:
        """Generate human-readable intervention description."""
        symptom = intervention["symptom"].replace("_", " ").title()
        category = intervention["intervention_category"]

        descriptions = {
            "Lifestyle": f"Lifestyle modifications to address {symptom}",
            "Medication": f"Medication to treat {symptom}",
            "Physical Therapy": f"Physical therapy for {symptom}",
            "Surgical": f"Surgical evaluation for {symptom}",
            "General Medical": f"Medical management of {symptom}"
        }

        return descriptions.get(category, f"Treatment targeting {symptom}")

    def _get_medical_disclaimer(self) -> List[str]:
        """Standard medical disclaimers for AI-generated suggestions."""
        return [
            "These suggestions are based on probabilistic modeling and are for educational purposes only",
            "Always consult with healthcare professionals before making treatment decisions",
            "Individual patient factors not captured in this model may affect treatment appropriateness",
            "This tool does not replace clinical judgment or medical expertise"
        ]

