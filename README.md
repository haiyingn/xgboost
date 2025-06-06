# Credit Card Fraud Detection

This project uses XGBoost and SHAP to detect fraudulent credit card transactions using machine learning.

## Overview

- Data preprocessing and binning
- Model build
- Evaluation using ROC-AUC and confusion matrix
- SHAP values for explainability

## Files

- `main.py`: Main script
- `creditcard.csv`: Dataset (if small; else download from Kaggle)
- `requirements.txt`: Dependencies

## Dataset

[Kaggle Dataset: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## How to Run

1. Clone the repo
2. Add `creditcard.csv` to the project root
3. Run:

```bash
pip install -r requirements.txt
python main.py
```

## License

MIT
