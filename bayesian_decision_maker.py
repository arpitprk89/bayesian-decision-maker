"""
Bayesian Decision Maker — Arpit Pareek
github.com/arpitprk89

A cognitive AI model that makes decisions under uncertainty
using Bayesian inference — inspired by how the human brain
updates beliefs and selects actions based on evidence.

Core concepts:
- Bayesian Belief Updating (prior → likelihood → posterior)
- Expected Utility Maximization
- Uncertainty Quantification
- Multi-hypothesis reasoning

Inspired by: Prof. Kenji Doya's research on
Bayesian Brain and Reinforcement Learning at OIST.
"""

import math
import random
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
#   CORE: BAYESIAN BELIEF SYSTEM
# ─────────────────────────────────────────────

class BayesianBelief:
    """
    Models a probability distribution over hypotheses.
    Updates beliefs using Bayes' Theorem:
        P(H|E) = P(E|H) * P(H) / P(E)
    """

    def __init__(self, hypotheses: Dict[str, float]):
        """
        Args:
            hypotheses: {hypothesis_name: prior_probability}
        Priors must sum to 1.0
        """
        total = sum(hypotheses.values())
        self.beliefs = {h: p / total for h, p in hypotheses.items()}
        self.history = [dict(self.beliefs)]  # Track belief evolution

    def update(self, likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Update beliefs given new evidence.
        likelihoods: P(evidence | hypothesis) for each hypothesis
        Returns updated posterior probabilities.
        """
        # Compute unnormalized posteriors
        posteriors = {}
        for h, prior in self.beliefs.items():
            likelihood = likelihoods.get(h, 1.0)
            posteriors[h] = prior * likelihood

        # Normalize — P(evidence) = sum of all unnormalized posteriors
        p_evidence = sum(posteriors.values())
        if p_evidence == 0:
            return self.beliefs  # No update if evidence impossible

        self.beliefs = {h: p / p_evidence for h, p in posteriors.items()}
        self.history.append(dict(self.beliefs))
        return self.beliefs

    def most_likely(self) -> Tuple[str, float]:
        """Return the hypothesis with highest posterior probability."""
        best = max(self.beliefs, key=lambda h: self.beliefs[h])
        return best, self.beliefs[best]

    def entropy(self) -> float:
        """
        Shannon entropy — measures uncertainty.
        High entropy = high uncertainty.
        Low entropy = confident belief.
        H = -sum(P(h) * log2(P(h)))
        """
        h = 0.0
        for p in self.beliefs.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def uncertainty_level(self) -> str:
        """Categorize uncertainty from entropy."""
        e = self.entropy()
        max_entropy = math.log2(len(self.beliefs)) if len(self.beliefs) > 1 else 1
        ratio = e / max_entropy if max_entropy > 0 else 0

        if ratio < 0.2:
            return "VERY LOW — Highly Confident"
        elif ratio < 0.4:
            return "LOW — Fairly Confident"
        elif ratio < 0.6:
            return "MEDIUM — Uncertain"
        elif ratio < 0.8:
            return "HIGH — Very Uncertain"
        else:
            return "VERY HIGH — Almost No Information"


# ─────────────────────────────────────────────
#   CORE: DECISION ENGINE
# ─────────────────────────────────────────────

class DecisionEngine:
    """
    Selects optimal actions using Expected Utility Maximization.
    EU(action) = sum over hypotheses of P(hypothesis) * Utility(action, hypothesis)
    
    This mirrors how the Bayesian Brain model suggests humans
    select actions — by integrating uncertainty with expected outcomes.
    """

    def __init__(self, actions: List[str], utility_matrix: Dict[str, Dict[str, float]]):
        """
        Args:
            actions: list of possible actions
            utility_matrix: {action: {hypothesis: utility_value}}
        """
        self.actions = actions
        self.utility_matrix = utility_matrix

    def expected_utility(self, action: str, beliefs: Dict[str, float]) -> float:
        """Compute EU(action) = Σ P(h) * U(action, h)"""
        eu = 0.0
        for hypothesis, prob in beliefs.items():
            utility = self.utility_matrix.get(action, {}).get(hypothesis, 0.0)
            eu += prob * utility
        return eu

    def best_action(self, beliefs: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns optimal action, its expected utility,
        and all EU values for comparison.
        """
        eu_values = {}
        for action in self.actions:
            eu_values[action] = self.expected_utility(action, beliefs)

        best = max(eu_values, key=eu_values.get)
        return best, eu_values[best], eu_values


# ─────────────────────────────────────────────
#   CORE: COGNITIVE AGENT
# ─────────────────────────────────────────────

class BayesianAgent:
    """
    A cognitive agent that perceives evidence, updates beliefs,
    and makes decisions — modelling aspects of human cognition
    under uncertainty.
    
    Inspired by the Free Energy Principle and Bayesian Brain
    hypothesis explored in computational neuroscience.
    """

    def __init__(self, name: str, belief: BayesianBelief, engine: DecisionEngine):
        self.name = name
        self.belief = belief
        self.engine = engine
        self.decision_log = []
        self.evidence_count = 0

    def perceive(self, evidence_name: str, likelihoods: Dict[str, float]) -> Dict[str, float]:
        """Receive evidence and update beliefs."""
        self.evidence_count += 1
        updated = self.belief.update(likelihoods)
        return updated

    def decide(self) -> Tuple[str, float]:
        """Make optimal decision given current beliefs."""
        action, eu, all_eu = self.engine.best_action(self.belief.beliefs)
        self.decision_log.append({
            'beliefs': dict(self.belief.beliefs),
            'action': action,
            'expected_utility': eu,
            'all_eu': all_eu,
            'uncertainty': self.belief.entropy()
        })
        return action, eu

    def full_report(self) -> str:
        """Generate a detailed cognitive state report."""
        best_h, best_p = self.belief.most_likely()
        action, eu, all_eu = self.engine.best_action(self.belief.beliefs)

        lines = [
            f"\n{'═'*60}",
            f"  BAYESIAN AGENT: {self.name}",
            f"{'═'*60}",
            f"\n  CURRENT BELIEFS (Posterior Probabilities):",
        ]

        sorted_beliefs = sorted(self.belief.beliefs.items(), key=lambda x: -x[1])
        for h, p in sorted_beliefs:
            bar = '█' * int(p * 30)
            lines.append(f"    {h:<25} {bar:<30} {p:.4f} ({p*100:.1f}%)")

        lines += [
            f"\n  UNCERTAINTY ANALYSIS:",
            f"    Shannon Entropy:  {self.belief.entropy():.4f} bits",
            f"    Uncertainty Level: {self.belief.uncertainty_level()}",
            f"    Evidence Received: {self.evidence_count} observations",
            f"\n  DECISION ANALYSIS (Expected Utility):",
        ]

        sorted_eu = sorted(all_eu.items(), key=lambda x: -x[1])
        for act, eu_val in sorted_eu:
            marker = " ◄ OPTIMAL" if act == action else ""
            lines.append(f"    {act:<25} EU = {eu_val:+.4f}{marker}")

        lines += [
            f"\n  CONCLUSION:",
            f"    Most likely state:  {best_h} ({best_p*100:.1f}% confidence)",
            f"    Optimal action:     {action}",
            f"    Expected utility:   {eu:.4f}",
            f"{'═'*60}\n",
        ]

        return "\n".join(lines)


# ─────────────────────────────────────────────
#   SCENARIO 1: MEDICAL DIAGNOSIS
#   (Inspired by real Bayesian applications in healthcare)
# ─────────────────────────────────────────────

def scenario_medical_diagnosis():
    print("\n" + "="*60)
    print("  SCENARIO 1: MEDICAL DIAGNOSIS UNDER UNCERTAINTY")
    print("  Modelling clinical reasoning with Bayesian inference")
    print("="*60)

    # Prior probabilities (base rates from epidemiology)
    belief = BayesianBelief({
        "Flu":          0.40,   # Common
        "COVID-19":     0.25,   # Moderate prevalence
        "Common Cold":  0.25,   # Common
        "Pneumonia":    0.10,   # Less common
    })

    # Treatment options
    engine = DecisionEngine(
        actions=["Antiviral + Rest", "Isolate + Monitor", "Antibiotics", "Symptomatic Relief"],
        utility_matrix={
            "Antiviral + Rest":     {"Flu": 0.9, "COVID-19": 0.5, "Common Cold": 0.3, "Pneumonia": -0.2},
            "Isolate + Monitor":    {"Flu": 0.4, "COVID-19": 0.9, "Common Cold": 0.2, "Pneumonia": 0.3},
            "Antibiotics":          {"Flu": -0.3, "COVID-19": -0.1, "Common Cold": -0.1, "Pneumonia": 0.95},
            "Symptomatic Relief":   {"Flu": 0.5, "COVID-19": 0.4, "Common Cold": 0.8, "Pneumonia": 0.1},
        }
    )

    agent = BayesianAgent("Medical Diagnostic AI", belief, engine)

    print("\n  Initial state — Before any tests:")
    print(agent.full_report())

    # Evidence 1: Fever detected
    print("  [EVIDENCE 1] Patient has high fever (38.9°C)")
    agent.perceive("High Fever", {
        "Flu": 0.85, "COVID-19": 0.75, "Common Cold": 0.30, "Pneumonia": 0.70
    })
    print(agent.full_report())

    # Evidence 2: Loss of taste/smell
    print("  [EVIDENCE 2] Patient reports loss of taste and smell")
    agent.perceive("Loss of Taste/Smell", {
        "Flu": 0.15, "COVID-19": 0.90, "Common Cold": 0.05, "Pneumonia": 0.10
    })
    print(agent.full_report())

    # Evidence 3: Oxygen level normal
    print("  [EVIDENCE 3] Oxygen saturation: 97% (normal)")
    agent.perceive("Normal O2", {
        "Flu": 0.85, "COVID-19": 0.70, "Common Cold": 0.95, "Pneumonia": 0.20
    })
    print(agent.full_report())


# ─────────────────────────────────────────────
#   SCENARIO 2: AUTONOMOUS NAVIGATION
# ─────────────────────────────────────────────

def scenario_autonomous_navigation():
    print("\n" + "="*60)
    print("  SCENARIO 2: AUTONOMOUS ROBOT NAVIGATION")
    print("  Bayesian localisation under sensor uncertainty")
    print("="*60)

    belief = BayesianBelief({
        "Zone_A_Library":    0.25,
        "Zone_B_Cafeteria":  0.25,
        "Zone_C_Lab":        0.25,
        "Zone_D_Corridor":   0.25,
    })

    engine = DecisionEngine(
        actions=["Go_North", "Go_South", "Go_East", "Go_West", "Request_Help"],
        utility_matrix={
            "Go_North":    {"Zone_A_Library": 0.8, "Zone_B_Cafeteria": -0.3, "Zone_C_Lab": 0.5, "Zone_D_Corridor": 0.2},
            "Go_South":    {"Zone_A_Library": -0.2, "Zone_B_Cafeteria": 0.9, "Zone_C_Lab": -0.1, "Zone_D_Corridor": 0.3},
            "Go_East":     {"Zone_A_Library": 0.3, "Zone_B_Cafeteria": 0.2, "Zone_C_Lab": 0.8, "Zone_D_Corridor": 0.6},
            "Go_West":     {"Zone_A_Library": 0.4, "Zone_B_Cafeteria": 0.3, "Zone_C_Lab": 0.2, "Zone_D_Corridor": 0.7},
            "Request_Help": {"Zone_A_Library": 0.5, "Zone_B_Cafeteria": 0.5, "Zone_C_Lab": 0.5, "Zone_D_Corridor": 0.5},
        }
    )

    agent = BayesianAgent("Navigation AI", belief, engine)

    print("\n  Initial state — Robot powered on, location unknown:")
    print(agent.full_report())

    # Sensor reading 1
    print("  [SENSOR 1] Detects: Bookshelves and quiet environment")
    agent.perceive("Bookshelves+Quiet", {
        "Zone_A_Library": 0.90, "Zone_B_Cafeteria": 0.05,
        "Zone_C_Lab": 0.15, "Zone_D_Corridor": 0.10
    })
    print(agent.full_report())

    # Sensor reading 2
    print("  [SENSOR 2] Detects: No food smell, low ambient noise")
    agent.perceive("No Food, Low Noise", {
        "Zone_A_Library": 0.85, "Zone_B_Cafeteria": 0.02,
        "Zone_C_Lab": 0.30, "Zone_D_Corridor": 0.20
    })
    print(agent.full_report())


# ─────────────────────────────────────────────
#   SCENARIO 3: COGNITIVE EMOTION RECOGNITION
#   (Connected to Arpit's own cognitive-ai-simulator)
# ─────────────────────────────────────────────

def scenario_emotion_recognition():
    print("\n" + "="*60)
    print("  SCENARIO 3: BAYESIAN EMOTION RECOGNITION")
    print("  Cognitive model of emotional state estimation")
    print("  (Extension of cognitive-ai-simulator project)")
    print("="*60)

    belief = BayesianBelief({
        "Happy":   0.20,
        "Sad":     0.20,
        "Anxious": 0.20,
        "Angry":   0.20,
        "Neutral": 0.20,
    })

    engine = DecisionEngine(
        actions=["Offer Support", "Engage Playfully", "Give Space", "Stay Calm", "Ask Questions"],
        utility_matrix={
            "Offer Support":   {"Happy": 0.2, "Sad": 0.9, "Anxious": 0.8, "Angry": 0.3, "Neutral": 0.3},
            "Engage Playfully":{"Happy": 0.9, "Sad": 0.1, "Anxious": 0.2, "Angry": -0.3, "Neutral": 0.5},
            "Give Space":      {"Happy": 0.3, "Sad": 0.4, "Anxious": 0.5, "Angry": 0.9, "Neutral": 0.4},
            "Stay Calm":       {"Happy": 0.4, "Sad": 0.5, "Anxious": 0.7, "Angry": 0.6, "Neutral": 0.7},
            "Ask Questions":   {"Happy": 0.5, "Sad": 0.6, "Anxious": 0.4, "Angry": 0.2, "Neutral": 0.8},
        }
    )

    agent = BayesianAgent("Emotion Recognition AI", belief, engine)

    print("\n  Initial state — No signals observed:")
    print(agent.full_report())

    print("  [SIGNAL 1] Voice pitch: Low and slow")
    agent.perceive("Low Slow Voice", {
        "Happy": 0.10, "Sad": 0.85, "Anxious": 0.40, "Angry": 0.20, "Neutral": 0.30
    })
    print(agent.full_report())

    print("  [SIGNAL 2] Facial expression: Downturned mouth, eyes cast down")
    agent.perceive("Sad Face", {
        "Happy": 0.02, "Sad": 0.92, "Anxious": 0.20, "Angry": 0.15, "Neutral": 0.10
    })
    print(agent.full_report())

    print("  [SIGNAL 3] Response time: Slow, short answers")
    agent.perceive("Slow Short Answers", {
        "Happy": 0.05, "Sad": 0.80, "Anxious": 0.50, "Angry": 0.30, "Neutral": 0.25
    })
    print(agent.full_report())


# ─────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  BAYESIAN DECISION MAKER — ARPIT PAREEK")
    print("  Modelling intelligence under uncertainty")
    print("  github.com/arpitprk89/bayesian-decision-maker")
    print("  Inspired by: Prof. Kenji Doya, OIST Neural Computation Unit")
    print("█"*60)

    scenario_medical_diagnosis()
    scenario_autonomous_navigation()
    scenario_emotion_recognition()

    print("\n" + "█"*60)
    print("  CORE CONCEPTS DEMONSTRATED:")
    print("  1. Bayes' Theorem: P(H|E) = P(E|H)*P(H) / P(E)")
    print("  2. Belief Updating: Prior → Evidence → Posterior")
    print("  3. Shannon Entropy: Quantifying uncertainty")
    print("  4. Expected Utility: EU(a) = Σ P(h)*U(a,h)")
    print("  5. Optimal Decision: argmax EU(action)")
    print()
    print("  CONNECTION TO NEUROSCIENCE:")
    print("  The Bayesian Brain hypothesis (Doya et al.) suggests")
    print("  the brain performs probabilistic inference — combining")
    print("  prior beliefs with sensory evidence to make decisions.")
    print("  This code models that process computationally.")
    print("█"*60 + "\n")
