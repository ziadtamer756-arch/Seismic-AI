# Experimental Results

## 1. Model Comparison

Multiple machine learning approaches were evaluated and compared using:

- Accuracy
- Precision
- Recall
- F1-score


## 2. Hyperparameter Optimization

Optimization experiments were performed to identify suitable model parameters.

The optimized parameters are stored in:

evaluation/hyperparameter_results.json


## 3. Feature Ablation Study

Ablation experiments evaluate the contribution of different feature groups:

- Full Features
- Without Historical Features
- Without Time Features
- Geographic Features Only


## 4. Statistical Validation

The system reports:

- Accuracy estimation
- 95% confidence interval


## 5. Neural Network Regularization

MLP experiments include:

- Hidden layer configuration
- Alpha regularization
- Early stopping evaluation

Results are stored in:

evaluation/mlp_tuning_results.json
