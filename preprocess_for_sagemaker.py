import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
import joblib
import os

# Load data
FILE = "data_files/credit_risk_with_targets_cleaned_final.csv"
df = pd.read_csv(FILE)

# Step 1: Fill missing values
df["Saving accounts"].fillna("unknown", inplace=True)
df["Checking account"].fillna("unknown", inplace=True)

# Step 2: Extract features and labels (No description)
X = df[["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose"]]
y = df["target"].map({"good risk": 1, "bad risk": 0})  # Binary label

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Define column types
categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
numerical_cols = ["Age", "Credit amount", "Duration"]

# Step 5: Define and fit preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols),
])

X_train_final = preprocessor.fit_transform(X_train)
X_test_final = preprocessor.transform(X_test)

# Step 6: Save transformers
os.makedirs("artifacts", exist_ok=True)
joblib.dump(preprocessor, "artifacts/tabular_preprocessor.joblib")

# Step 7: Export CSVs (target in last column, no header)
train_df = pd.DataFrame(X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final)
train_df["target"] = y_train.values
train_df.to_csv("train_data_sagemaker.csv", index=False, header=False)

test_df = pd.DataFrame(X_test_final.toarray() if hasattr(X_test_final, "toarray") else X_test_final)
test_df["target"] = y_test.values
test_df.to_csv("test_data_sagemaker.csv", index=False, header=False)

# Step 8: Save true labels for AUC evaluation
y_test.to_csv("y_test_true_labels.csv", index=False)

print("✅ Preprocessing complete. Saved:")
print("- train_data_sagemaker.csv")
print("- test_data_sagemaker.csv")
print("- y_test_true_labels.csv")
print("- artifacts/tabular_preprocessor.joblib")