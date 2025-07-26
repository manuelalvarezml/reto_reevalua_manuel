# train_logreg.py
import pandas as pd
import joblib
import os
import shutil
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
    joblib.dump(model, f"{output_path}/model.joblib")

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
def model_fn(model_dir):
    """Load the trained model"""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse input data from the request"""
    if request_content_type == "text/csv":
        import io
        return pd.read_csv(io.StringIO(request_body), header=None)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Generate predictions"""
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, response_content_type):
    """Format predictions as CSV response"""
    if response_content_type == "text/csv":
        return ",".join(str(x) for x in prediction)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")