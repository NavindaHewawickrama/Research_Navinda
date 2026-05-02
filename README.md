Interpretable Deep Learning for Robust Drug Mechanism Prediction in High-Throughput Gene Expression Data Navinda Hewawickrama & Sugandima Vidanagamachchi April 2025

Repo created on October 22, 2025

Decoding the precise mechanisms of action (MoA) of small-molecule compounds re
mains a formidable challenge in modern drug discovery and repurposing. While
high-throughput transcriptomic resources, such as the Library of Integrated Network
Based Cellular Signatures (LINCS) L1000 dataset, offer unprecedented opportunities
to profile cellular responses to chemical perturbations at scale, translating this raw
data into actionable biological insights is notoriously difficult. Computational analy
sis is frequently hindered by the curse of high dimensionality, pervasive experimental
noise, and the inherently limited interpretability of complex machine learning archi
tectures.
This thesis systematically investigates computational methodologies for predict
ing drug MoA from LINCS L1000 gene expression profiles, shifting the analytical
focus from mere predictive accuracy to the fundamental reliability of Explainable AI
(XAI). Specifically, we interrogate the stability of post-hoc feature attributions, using
SHapley Additive exPlanations (SHAP), under conditions of severe class imbalance.
Through a series of rigorously controlled experiments evaluating knowledge-guided
feature representations, we demonstrate that explanatory stability is not a static
property of an algorithm, but is instead critically dependent on class granularity and
baseline predictive competence.
We reveal that under an extreme 110-class formulation (characterized by a Macro
F1 ≈0.006), SHAP explanations degenerate into statistical illusions. Across indepen
dent random initializations, the models produced effectively random feature rankings,
yielding a mean Jaccard similarity of merely 0.078. Conversely, when the prediction
task is refined into a biologically coherent 24-class structure, predictive performance
crosses a critical threshold (Macro F1 ≈ 0.19–0.30). In this optimized regime, both
stochastic ensemble models (Random Forest) and deterministic linear models (Logis
tic Regression) achieve perfect explanatory reproducibility (Jaccard = 1.000).
Ultimately, these findings establish the first quantitative benchmark of SHAP sta
bility across varying class granularities in multi-class MoA prediction. This research
exposes a vital vulnerability in computational pharmacogenomics: the reliability of
XAI collapses entirely when model performance falls below a critical threshold. By
providing a methodological framework to validate explanation robustness, this work
underscores the profound necessity for skepticism and rigorous stability testing when
interpreting post-hoc explanations in high-dimensional, highly imbalanced transcrip
tomic environments.

Google colab ntebooks

BaselineML Experiments - https://colab.research.google.com/drive/19-v-soqI99LDiq35OkXA2PVKu-0otX?usp=sharing

MSigDBHallmark+RF - https://colab.research.google.com/drive/15ayOwz4nHWB4I6yXXs1dtRXnbZUpYaaE?usp=sharing

HybridFeatureModel - https://colab.research.google.com/drive/1WvwIuQyNHS29FWog5jQd5ky91i7AxNx?usp=sharing

MLP Hybrid Deep Learning - https://colab.research.google.com/drive/1l1KoHMOZEhXnrKA3IzOaSgcLc7GC-yW?usp=sharing

Autoencoder+Classifier - https://colab.research.google.com/drive/11RtXQlnDmAN8yAw5xxL3LyP16NGdTCI?usp=sharing

Graph Embedding KG+HybridRF - https://colab.research.google.com/drive/1vzKsQaRUdM1jOI2Wgf03TUlu7imO4Ws?usp=sharing

KG + Supervised Feature Selection + SHAP - https://colab.research.google.com/drive/1BGd3ArduJiMx20v90WXuenxr83Zn0eh4?usp=sharing

SHAP Stability AcrossRandomSeeds - https://colab.research.google.com/drive/1nkivvJhlL2c-SRR9s52ETCpGFiUxhToA?usp=sharing

SHAPStability–Hy bridRF(24Class) -  https://colab.research.google.com/drive/1MsW1knk9-XmXpyYCYLgbBiHKKnmXSM3f?usp=sharing

110-ClassKG+Bal ancedRF - https://colab.research.google.com/drive/1TtRQcRCYW8BUpsMlBvSuJK88k6VCheGP?usp=sharing
