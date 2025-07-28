import boto3
import pandas as pd
import json

# === Load data ===
df = pd.read_csv("test_data_for_inference.csv")  # Use header from file
payload = df.to_csv(index=False, header=True)    # Header needed for column names

# === Set up SageMaker runtime client ===
runtime = boto3.client("sagemaker-runtime")
endpoint_name = "logreg-fraud-detector-manuel"

# === Invoke endpoint ===
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Accept="application/json",  # Important: expect JSON output
    Body=payload
)

# === Parse and display results ===
result = response["Body"].read().decode("utf-8")
predictions = json.loads(result)  # Parse string like "[1, 0, 1]"

df["prediction"] = predictions

print("✅ Predictions:")
print(df)