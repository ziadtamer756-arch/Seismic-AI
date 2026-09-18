\# SeisBench Baseline Experiment



\## Dataset



Dataset:

STEAD (Stanford Earthquake Dataset)



Source:

SeisBench / STEAD repository



\## Toolbox



SeisBench



Purpose:

Standardized framework for seismic machine learning models and pretrained waveform models.



\## Task



\- Earthquake Detection

\- P-wave Phase Picking

\- S-wave Phase Picking





\## Pretrained Models



The baseline uses multiple pretrained models from the same SeisBench toolbox:



\### 1. PhaseNet



Task:

Phase Picking



Output:

\- P phase arrival

\- S phase arrival





\### 2. EQTransformer



Task:

\- Earthquake Detection

\- Phase Picking



Output:

\- Detection probability

\- P arrival

\- S arrival





\### 3. GPD



Task:

Phase Picking



Output:

\- P/S phase probabilities





\## Evaluation Metrics



The baseline evaluation is based on:



\- Detection performance

\- P arrival time residual error

\- S arrival time residual error





\## Comparison



Baseline:



STEAD + SeisBench pretrained models





Proposed Model:



DNN + Wide Deep + TabNet + Ensemble AI





\## References



\- SeisBench: A Toolbox for Machine Learning in Seismology

\- Quantitative Evaluation of Deep Learning Based Seismic Pickers

