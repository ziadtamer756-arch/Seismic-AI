\# Baseline vs Proposed Model Comparison



\## 1. Baseline Approach



\### Dataset

STEAD (Stanford Earthquake Dataset)



\### Toolbox

SeisBench



\### Task

\- Earthquake Detection

\- P-wave Phase Picking

\- S-wave Phase Picking



\### Pretrained Models



| Model | Type | Task |

|---|---|---|

| PhaseNet | Pretrained SeisBench Model | P/S Phase Picking |

| EQTransformer | Pretrained SeisBench Model | Detection + Phase Picking |

| GPD | Pretrained SeisBench Model | Phase Picking |



\---



\# 2. Proposed Model



\## Objective



Develop an intelligent earthquake risk prediction system using machine learning and deep learning techniques.



\## Architecture



Input Data



↓



Feature Engineering



↓



Deep Learning Models:



\- DNN

\- Wide \& Deep Network

\- TabNet



↓



Ensemble AI (Soft Voting)



↓



Risk Prediction





\---



\# 3. Comparison



| Aspect | Baseline | Proposed Model |

|---|---|---|

| Dataset | STEAD | Earthquake Feature Dataset |

| Framework | SeisBench | TensorFlow + PyTorch |

| Models | PhaseNet, EQTransformer, GPD | DNN, Wide Deep, TabNet, Ensemble |

| Task | Detection and Phase Picking | Risk Classification |

| Output | P/S arrival picks and detection | Risk level and confidence score |





\---



\# 4. Contribution



The baseline provides comparison with existing pretrained seismic models.



The proposed model introduces an ensemble deep learning architecture combining multiple neural architectures to improve earthquake risk prediction performance.



