\# Experimental Results Summary



\## Baseline Results (SeisBench)



The baseline experiment uses pretrained seismic models from the SeisBench toolbox on STEAD-related phase picking and earthquake detection tasks.



| Model | Dataset | Task | Evaluation Metric |

|---|---|---|---|

| PhaseNet | STEAD | P/S Phase Picking | Arrival time residual error |

| EQTransformer | STEAD | Earthquake Detection + P/S Picking | Detection performance + arrival residual error |

| GPD | STEAD | P/S Phase Picking | Arrival time residual error |



Reference:

Münchmeyer et al. "Which Picker Fits My Data? A Quantitative Evaluation of Deep Learning Based Seismic Pickers"



\---



\## Proposed Model Results



| Model | Task | Accuracy | F1 Score |

|---|---|---|---|

| DNN | Earthquake Classification | - | - |

| Wide \& Deep | Earthquake Classification | - | - |

| TabNet | Earthquake Classification | - | - |

| Ensemble AI | Earthquake Classification | 92.33% | 91.58% |



\---



\## Final Comparison



| Component | Baseline | Proposed |

|---|---|---|

| Framework | SeisBench | TensorFlow / PyTorch |

| Models | PhaseNet, EQTransformer, GPD | DNN, Wide Deep, TabNet Ensemble |

| Main Task | Detection and Phase Picking | Earthquake Risk Prediction |

| Output | P/S arrival picks and detection | Risk level and confidence |



