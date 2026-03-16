# ECG Classification: Related Work and Task Definitions

## Task Definitions Used in This Reproduction

### Single-Label Classification

Each 12-lead ECG recording is assigned exactly one diagnostic class by taking the
argmax of its multi-hot label vector. The model outputs a softmax distribution over
9 classes and is trained with cross-entropy loss.

This formulation is a simplification: approximately 6.9% of CPSC2018 records carry
two or three simultaneous diagnoses. Taking argmax discards the secondary labels,
keeping only the numerically first active class. The simplification is reasonable as
a proof-of-concept because the vast majority of records are single-label, and it
allows direct use of standard multi-class infrastructure (softmax + cross-entropy).

However, argmax-of-multi-hot is **not standard practice** in the ECG literature.
Where single-label formulations are used, the underlying data is typically
inherently single-label (e.g., rhythm classification at a specific time point).

### Multi-Label Classification

Each recording can have multiple simultaneous diagnoses (multi-hot encoding). The
model outputs independent logits per class and is trained with binary cross-entropy
(BCE) loss. Evaluation uses macro F1 computed from per-class precision and recall.

This is the formulation aligned with how CPSC2018 was originally designed and how
major challenges (CinC 2020/2021) and benchmark papers (PTB-XL, Ribeiro et al.)
frame the ECG classification task.

## How the Source Datasets Were Originally Framed

### CPSC2018 (China Physiological Signal Challenge 2018)

The original challenge was framed as **multi-label classification**. Out of 6,877
recordings across 9 classes (SNR, AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE),
approximately 476 (~6.9%) carry two or three labels. The scoring metric was the
average of nine per-class F1 scores (macro F1). For multi-labeled recordings,
predicting any one of the true labels counted as a true positive -- a lenient
scoring scheme that did not strictly penalize missing co-occurring conditions.

- CPSC2018 Official Challenge: http://2018.icbeb.org/Challenge.html
- Chen, Huang, Shih, Hu & Hwang (2020). "Detection and Classification of Cardiac
  Arrhythmias by a Challenge-Best Deep Learning Neural Network Model." *iScience*,
  23(3).

### PhysioNet/CinC 2020 and 2021

Both challenges were **explicitly multi-label classification** tasks.

**CinC 2020** scored 27 diagnoses across 66,361 12-lead ECGs from six hospital
systems in four countries. Each algorithm had to output a set of one or more classes
with confidence scores. The challenge introduced a clinically-weighted accuracy
metric that awards partial credit for misdiagnoses resulting in similar treatments.
Most top teams used sigmoid activation + BCE loss with class weighting.

**CinC 2021 ("Will Two Do?")** extended the 2020 challenge to varying lead
configurations (12, 6, 4, 3, and 2 leads), classifying 30 cardiac abnormalities.
CPSC2018 data was included as one of its constituent databases. The same
clinically-weighted scoring metric was used.

- Perez Alday et al. (2021). "Classification of 12-lead ECGs: the
  PhysioNet/Computing in Cardiology Challenge 2020." *Physiological Measurement*,
  41(12).
- Reyna et al. (2022). "Will Two Do? Varying Dimensions in Electrocardiography:
  The PhysioNet/Computing in Cardiology Challenge 2021."

## Key Papers: Single-Label ECG Classification

**Hannun et al. (2019)** used a single-label softmax formulation to classify each
segment of single-lead ambulatory (Holter) ECGs into one of 12 rhythm classes.
Single-label was appropriate here because the task was rhythm classification (what
rhythm is happening at this moment), where mutual exclusivity is a reasonable
assumption. Architecture: 33 convolutional layers + linear + softmax. Average AUC
0.97, F1 0.837.

- Hannun et al. (2019). "Cardiologist-level arrhythmia detection and classification
  in ambulatory electrocardiograms using a deep neural network." *Nature Medicine*.

**MIT-BIH Arrhythmia Database studies** -- the vast majority of papers using this
database follow AAMI standards and treat heartbeat classification as a 5-class
single-label problem (N, SVEB, VEB, F, Q) using softmax + categorical
cross-entropy. This is the most common single-label ECG benchmark.

## Key Papers: Multi-Label ECG Classification

**Ribeiro et al. (2020)** trained on >2 million 12-lead ECGs. Used sigmoid
activation because "the classes are not mutually exclusive (i.e., two or more
classes may occur in the same exam)." Six abnormalities classified simultaneously.
F1 scores above 80%.

- Ribeiro et al. (2020). "Automatic diagnosis of the 12-lead ECG using a deep
  neural network." *Nature Communications*.

**Natarajan et al. (2020)** -- winner of CinC 2020 (team prna, score 0.533).
Combined hand-crafted features with transformer representations for multi-label
classification of 27 cardiac abnormalities.

- Natarajan et al. (2020). "A Wide and Deep Transformer Neural Network for 12-Lead
  ECG Classification." *Computing in Cardiology 2020*.

**Strodthoff et al. (2020)** established the PTB-XL benchmark (21,837 records) with
multi-label evaluation using macro AUROC as the primary metric (0.93 for diagnostic
superclass). This has become a standard ECG benchmarking reference.

- Strodthoff et al. (2020). "Deep Learning for ECG Analysis: Benchmarks and
  Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics*.

**He et al. (2021)** explicitly argued that "the study of multi-label ECG signal
classification is more important than the study of single-label ECG signal
classification" and proposed cost-sensitive thresholding for multi-label ECG.

- He et al. (2021). "Automatic Multi-Label ECG Classification with Category
  Imbalance and Cost-Sensitive Thresholding." *Biosensors*.

## Clinical Relevance

Multi-label is considered more clinically relevant by the research community:

- Cardiac conditions commonly co-occur (e.g., AF + RBBB, STD + I-AVB).
- The CinC 2020/2021 challenges were specifically designed around multi-label
  because "ECG classification is, by nature, a multi-label classification task."
- The cost-sensitive nature of ECG diagnosis (different misclassifications have
  different clinical impacts) is better captured in multi-label frameworks.
- Ribeiro et al. explicitly chose sigmoid over softmax because conditions are not
  mutually exclusive.

Single-label is appropriate when the task is inherently single-label (rhythm
classification, MIT-BIH heartbeat classification) or as a simplified
proof-of-concept.

## Standard Evaluation Metrics

| Metric | Where Used | Notes |
|--------|-----------|-------|
| Macro F1 | CPSC2018, many papers | Average of per-class F1; standard for imbalanced data |
| Macro AUROC | PTB-XL benchmark, general | Threshold-free; primary metric in PTB-XL |
| Clinically-weighted accuracy | CinC 2020/2021 | Custom metric with cardiologist-designed weight matrix |
| Per-class sensitivity/specificity | Ribeiro et al. 2020, clinical | Specificity >99% was a key result |
| Sample-level F1 | Multi-label papers | F1 computed per-sample then averaged |

This reproduction uses **macro F1** as the primary metric, consistent with the
CPSC2018 challenge and the majority of ECG classification papers.

## Summary of Benchmark Papers

| Paper | Year | Venue | Formulation | Dataset |
|-------|------|-------|-------------|---------|
| Hannun et al. | 2019 | Nature Medicine | Single-label (12 classes) | 91K single-lead Holter ECGs |
| Ribeiro et al. | 2020 | Nature Comms | Multi-label (6 abnormalities) | 2.3M 12-lead ECGs |
| Chen & Huang | 2020 | iScience | Multi-label (CPSC2018 winner) | CPSC2018 (6,877 records) |
| Strodthoff et al. | 2020 | IEEE JBHI | Multi-label (macro AUC) | PTB-XL (21,837 records) |
| Perez Alday et al. | 2021 | Physiol. Meas. | Multi-label (27 classes) | CinC 2020 (66,361 records) |
| He et al. | 2021 | Biosensors | Multi-label + cost-sensitive | CPSC2018 |
| Hong et al. | 2022 | Front. Physiol. | Meta-analysis of CinC 2020 | CinC 2020 |
