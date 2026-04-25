# Bayesian Decision Maker
### Modelling Intelligence Under Uncertainty

**By Arpit Pareek** | [github.com/arpitprk89](https://github.com/arpitprk89) | [Portfolio](https://arpitprk89.github.io/arpit-portfolio)

---

## What is This?

A cognitive AI system that makes decisions under uncertainty using **Bayesian inference** — modelling how the human brain updates beliefs and selects optimal actions based on evidence.

This project is directly inspired by the **Bayesian Brain hypothesis** explored in computational neuroscience, particularly the work of **Professor Kenji Doya** at the Neural Computation Unit, Okinawa Institute of Science and Technology (OIST), Japan.

---

## Core Concepts

### 1. Bayes' Theorem
```
P(Hypothesis | Evidence) = P(Evidence | Hypothesis) × P(Hypothesis) / P(Evidence)
```
The agent starts with prior beliefs, receives evidence, and computes updated posterior beliefs.

### 2. Shannon Entropy — Measuring Uncertainty
```
H = -Σ P(h) × log₂(P(h))
```
High entropy = high uncertainty. As evidence accumulates, entropy decreases — the agent becomes more confident.

### 3. Expected Utility Maximization
```
EU(action) = Σ P(hypothesis) × Utility(action, hypothesis)
```
The agent selects the action with the highest expected utility — integrating uncertainty with potential outcomes.

---

## Three Scenarios

### Scenario 1: Medical Diagnosis
The agent diagnoses a patient (Flu, COVID-19, Common Cold, Pneumonia) and selects the optimal treatment — updating beliefs as test results arrive.

### Scenario 2: Autonomous Robot Navigation
A robot localises itself in an unknown environment using sensor readings, updating its positional beliefs and selecting the optimal direction.

### Scenario 3: Bayesian Emotion Recognition
Extends my earlier **cognitive-ai-simulator** project with probabilistic inference — estimating emotional state from voice, facial expression, and response patterns, then selecting the optimal response strategy.

---

## Example Output

```
BAYESIAN AGENT: Medical Diagnostic AI

CURRENT BELIEFS:
  Flu              ████████████████  50.6%
  COVID-19         ████████          27.9%
  Common Cold      ███               11.2%
  Pneumonia        ███               10.4%

UNCERTAINTY ANALYSIS:
  Shannon Entropy:   1.70 bits
  Uncertainty Level: HIGH — Very Uncertain

[EVIDENCE] Loss of taste and smell detected...

UPDATED BELIEFS:
  COVID-19         █████████████████████  73.2%  ← Belief updated!
  Flu              ██████                 22.1%
  ...

OPTIMAL ACTION: Isolate + Monitor  (EU = 0.7891)
```

---

## Connection to Neuroscience

The **Bayesian Brain hypothesis** (Doya, Knill, Pouget et al.) proposes that the brain is fundamentally a probabilistic inference machine — it maintains internal models of the world, updates them with sensory evidence, and selects actions to minimise prediction error (Free Energy Principle, Friston).

This code computationally models that process:
- **Priors** = what the brain expects before evidence
- **Likelihood** = how strongly evidence supports each hypothesis  
- **Posterior** = updated belief after integrating evidence
- **Decision** = action selected to maximise expected outcome

---

## How to Run

```bash
git clone https://github.com/arpitprk89/bayesian-decision-maker
cd bayesian-decision-maker
python bayesian_decision_maker.py
```

No external libraries required — pure Python 3.

---

## My Other Projects

| Project | Description |
|---|---|
| [cognitive-ai-simulator](https://github.com/arpitprk89/cognitive-ai-simulator) | Emotion detection, memory retention, decision-making |
| [ai-quiz-game](https://github.com/arpitprk89/ai-quiz-game) | Adaptive human-AI interaction |
| [student-study-helper](https://github.com/arpitprk89/student-study-helper) | Data analysis for underserved students |
| [RL Visualizer](https://arpitprk89.github.io/turbo-fiesta/reinforcement_learning_visualizer.html) | Interactive Q-learning visualisation |

---

*Self-taught AI developer, 16 years old, Shrimadhopur, Rajasthan, India.*  
*No formal training. No lab. Just curiosity.*
