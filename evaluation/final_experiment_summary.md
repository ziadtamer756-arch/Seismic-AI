\# Final Experiment Summary



\## Project: Seismic-AI



\---



\# 1. Baseline Experiment



\## Objective



Evaluate existing pretrained seismic models using the SeisBench toolbox.



\## Dataset



STEAD (Stanford Earthquake Dataset)



\## Toolbox



SeisBench



\## Tasks



\- Earthquake Detection

\- P-wave Phase Picking

\- S-wave Phase Picking





\## Baseline Models



| Model | Type | Task |

|---|---|---|

| PhaseNet | Pretrained Model | P/S Phase Picking |

| EQTransformer | Pretrained Model | Detection + Phase Picking |

| GPD | Pretrained Model | Phase Picking |





\---



\# 2. Proposed Model



\## Objective



Develop an ensemble deep learning system for earthquake risk prediction.



\## Architecture



Input Data



↓



Feature Engineering



↓



Deep Learning Models:



\- DNN

\- Wide \& Deep

\- TabNet



↓



Soft Voting Ensemble



↓



Prediction





\---



\# 3. Proposed Contributions



\- Multi-model deep learning ensemble.

\- Combining different neural architectures.

\- Real-time prediction module.

\- Explainable AI using SHAP.

\- Interactive Streamlit dashboard.





\---



\# 4. Comparison



| Aspect | Baseline | Proposed |

|---|---|---|

| Framework | SeisBench | TensorFlow/PyTorch |

| Models | PhaseNet, EQTransformer, GPD | DNN, Wide Deep, TabNet |

| Task | Detection and Phase Picking | Risk Prediction |

| Output | P/S picks and detection | Risk level and confidence |





\---



\# 5. Project Status



Completed:



✅ SeisBench baseline setup  

✅ Pretrained model integration  

✅ Ensemble AI model  

✅ Real-time prediction module  

✅ Dashboard deployment  

✅ Documentation

