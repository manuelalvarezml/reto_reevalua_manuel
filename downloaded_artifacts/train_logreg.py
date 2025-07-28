import pandas as pd
import joblib
import logging
import os
import shutil
import json
import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

if __name__ == "__main__":
    # Environment paths from SageMaker
    input_train_path = os.environ["SM_CHANNEL_TRAIN"]
    input_test_path = os.environ.get("SM_CHANNEL_TEST", None)
    output_path = os.environ["SM_MODEL_DIR"]

    # Load training data
    train_df = pd.read_csv(f"{input_train_path}/train_data_sagemaker.csv", header=None)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"{output_path}/model.joblib", protocol=4)

    # === Evaluate if test data is available ===
    if input_test_path:
        try:
            test_df = pd.read_csv(f"{input_test_path}/test_data_sagemaker.csv", header=None)
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_prob)

            # Save metrics
            with open(f"{output_path}/evaluation.txt", "w") as f:
                f.write("Classification Report:\n")
                for label, metrics in report.items():
                    f.write(f"{label}: {metrics}\n")
                f.write(f"\nROC AUC Score: {auc:.4f}\n")

            # Save predictions to a temp location
            preds_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob
            })
            local_preds_path = "predictions.csv"
            preds_df.to_csv(local_preds_path, index=False)

            # Copy to SageMaker model output folder
            shutil.copy(local_preds_path, os.path.join(output_path, "predictions.csv"))

            print("✅ predictions.csv copied to output path.")
            print("✅ Evaluation and predictions saved.")

        except Exception as e:
            print("⚠️ Failed to evaluate on test data:", str(e))


# Inference functions for SageMaker hosting
def model_fn(model_dir="/opt/ml/model"):
    """Load the trained model"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(model_dir, "tabular_preprocessor.joblib"))
    return {"model": model, "preprocessor": preprocessor}

def input_fn(request_body, request_content_type):
    """Parse input data from the request"""
    if request_content_type == "text/csv":
        print("📥 Input data received:")
        print(request_body[:200]) # log first 200 chars
        return pd.read_csv(io.StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, loaded_artifacts):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    preprocessor = loaded_artifacts["preprocessor"]
    model = loaded_artifacts["model"]

    logger.info(f"📥 Input data received: {input_data.head()}")
    logger.info(f"📑 Columns received: {input_data.columns.tolist()}")

    try:
        # Step 1: Add bins (must match training)
        input_data = input_data.copy()
        input_data["Credit_bin"] = pd.qcut(input_data["Credit amount"], q=4, labels=["low", "mid-low", "mid-high", "high"])
        input_data["Age_bin"] = pd.qcut(input_data["Age"], q=4, labels=["young", "mid-young", "mid-old", "old"])
        input_data["Duration_bin"] = pd.qcut(input_data["Duration"], q=4, labels=["short", "mid-short", "mid-long", "long"])

        # Step 2: Select only the columns expected by the preprocessor
        input_data = input_data[
            ["Sex", "Job", "Housing", "Saving accounts", "Checking account",
             "Purpose", "Credit_bin", "Age_bin", "Duration_bin"]
        ]

        # Step 3: Transform
        input_transformed = preprocessor.transform(input_data)
        logger.info(f"✅ Successfully transformed input")
    except Exception as e:
        logger.error(f"❌ Failed during preprocessing or transform: {e}")
        raise e

    # Step 4: Predict
    return model.predict(input_transformed)

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        # If prediction is a NumPy array or pandas Series/DataFrame, convert it
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        return json.dumps(prediction), response_content_type

    elif response_content_type == "text/csv":
        import io
        csv_buffer = io.StringIO()
        if hasattr(prediction, "to_csv"):
            prediction.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue(), response_content_type
        else:
            raise ValueError("Prediction format not supported for CSV output")

    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")