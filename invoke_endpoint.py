import boto3
import pandas as pd
import json

# === Load data ===
df = pd.read_csv("test_data_for_inference.csv", header=None)
payload = df.to_csv(index=False, header=False)

# === Set up client ===
runtime = boto3.client("sagemaker-runtime")
endpoint_name = "logreg-fraud-detector-manuel"  # make sure this matches your deployed endpoint

# === Invoke endpoint ===
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload
)

# === Parse response ===
result = response["Body"].read().decode("utf-8")
predictions = [int(x) for x in result.strip().split(",")]
df["prediction"] = predictions

print("✅ Predictions:")
print(df)