# Credit Risk Classification with Generative AI + SageMaker

This project explores a hybrid approach to credit risk classification by combining traditional tabular data modeling with generative AI for feature enrichment (descriptions and labels). Final training is performed on AWS SageMaker using XGBoost.

## 📁 Project Structure
'''
├── artifacts/                   # Saved preprocessing transformers (joblib)
├── auxiliar_scripts/           # Error regeneration and unused helpers
├── bedrock_test_files/         # Scripts to test Bedrock generative models
├── data_files/                 # Raw and cleaned datasets
├── eda_credit_risk.ipynb       # Exploratory data analysis notebook
├── generate_descriptions.py    # Uses Bedrock to create descriptions per row
├── generate_risk_targets.py    # Uses Bedrock to classify risk from features
├── preprocess_for_sagemaker.py # Vectorizes + splits data for SageMaker training
├── train_model_sagemaker.py    # Launches training job on AWS SageMaker
├── train_data_sagemaker.csv    # Processed training set (no header)
├── test_data_sagemaker.csv     # Processed test set (no header)
├── y_test_true_labels.csv      # Ground truth labels for test set
'''
## 🚀 Pipeline Overview

1. **Generate Descriptions & Targets**:  
   Use Amazon Bedrock (`generate_descriptions.py`, `generate_risk_targets.py`) to generate richer features and labels.

2. **Preprocess for SageMaker**:  
   Run `preprocess_for_sagemaker.py` to encode tabular features, vectorize descriptions, and export train/test CSVs.

3. **Train Model on SageMaker**:  
   Launch training with `train_model_sagemaker.py` using XGBoost (built-in SageMaker container).

4. **Evaluate**:  
   Use `y_test_true_labels.csv` to assess predictions from the deployed model.

## 🧠 Goals

- Benchmark traditional tabular classification (e.g., XGBoost, logistic regression).
- Test generative AI (via Amazon Bedrock) for synthetic data labeling and feature enrichment.
- Deploy and scale via AWS SageMaker.

## ✅ Requirements

- Python 3.8+
- AWS CLI & SageMaker Python SDK
- Access to Amazon Bedrock (for description/label generation)

