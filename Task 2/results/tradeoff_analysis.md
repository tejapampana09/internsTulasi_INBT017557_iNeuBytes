### Accuracy-vs-Cost Comparison:
- **Best Classical Model**: Logistic Regression (TF-IDF 10k, Unigram) (10,001 params, 0.2s training)
- **Best LSTM Model**: LSTM (Dropout = 0.0) (531,553 params, 342.8s training)
- **Absolute F1-score Gain**: -10.04% points
- **Parameter Increase**: +0.5216M parameters
- **Training Time Increase**: +342.6 seconds
- **Verdict**: Classical model performs comparable to or better than LSTM while being vastly more efficient. Recommended to deploy the simpler classical model.
