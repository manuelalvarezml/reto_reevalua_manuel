import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
import pickletools
import pickle
import joblib
import os

# Load data
FILE = "data_files/credit_risk_with_targets_cleaned_final.csv"
df = pd.read_csv(FILE)

# Step 1: Fill missing values
df["Saving accounts"].fillna("unknown", inplace=True)
df["Checking account"].fillna("unknown", inplace=True)

# ✅ Step 2: Split first (on raw data)
df["target"] = df["target"].map({"good risk": 1, "bad risk": 0})  # Binary
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df.drop("target", axis=1),
    df["target"],
    test_size=0.2,
    stratify=df["target"],
    random_state=42
)

# ✅ Step 3: Transform train and test separately
def transform_raw_data(X):
    X = X.copy()
    X["Credit_bin"] = pd.qcut(X["Credit amount"], q=4, labels=["low", "mid-low", "mid-high", "high"])
    X["Age_bin"] = pd.qcut(X["Age"], q=4, labels=["young", "mid-young", "mid-old", "old"])
    X["Duration_bin"] = pd.qcut(X["Duration"], q=4, labels=["short", "mid-short", "mid-long", "long"])
    X.drop(columns=["Credit amount", "Age", "Duration"], inplace=True)
    return X[
        ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose", "Credit_bin", "Age_bin", "Duration_bin"]
    ]

X_train = transform_raw_data(X_train_raw)
X_test = transform_raw_data(X_test_raw)

# ✅ Step 4: Preprocessing (all categorical)
categorical_cols = X_train.columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

X_train_final = preprocessor.fit_transform(X_train)
X_test_final = preprocessor.transform(X_test)

# ✅ Step 5: Save artifacts
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/tabular_preprocessor.joblib", "wb") as f:
    pickler = pickle.Pickler(f, protocol=4)
    pickler.dump(preprocessor)

# Check protocol version of the saved file
with open("artifacts/tabular_preprocessor.joblib", "rb") as f:
    first_bytes = f.read(2)
    protocol_version = first_bytes[1] if first_bytes[0] == 0x80 else "unknown"

print(f"✅ Saved tabular_preprocessor.joblib using pickle protocol: {protocol_version}")

# ✅ Step 6: Save train/test files
os.makedirs("dataset_for_sagemaker/train", exist_ok=True)
os.makedirs("dataset_for_sagemaker/test", exist_ok=True)

train_df = pd.DataFrame(X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final)
train_df["target"] = y_train.values
train_df.to_csv("dataset_for_sagemaker/train/train_data_sagemaker.csv", index=False, header=False)

test_df = pd.DataFrame(X_test_final.toarray() if hasattr(X_test_final, "toarray") else X_test_final)
test_df["target"] = y_test.values
test_df.to_csv("dataset_for_sagemaker/test/test_data_sagemaker.csv", index=False, header=False)

# Save true test labels for AUC
y_test.to_csv("dataset_for_sagemaker/test/y_test_true_labels.csv", index=False)

print("✅ Preprocessing complete. Saved:")
print("- train/train_data_sagemaker.csv")
print("- test/test_data_sagemaker.csv")
print("- test/y_test_true_labels.csv")
print("- artifacts/tabular_preprocessor.joblib")