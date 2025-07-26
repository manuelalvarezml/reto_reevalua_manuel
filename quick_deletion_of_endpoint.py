import boto3

sagemaker = boto3.client("sagemaker", region_name="us-west-2")
endpoint_name = "logreg-fraud-detector-manuel"

# Delete endpoint
try:
    sagemaker.delete_endpoint(EndpointName=endpoint_name)
    print("🗑️ Deleted existing endpoint.")
except Exception as e:
    print("⚠️ Could not delete endpoint:", str(e))

# Delete endpoint config
try:
    sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("🗑️ Deleted existing endpoint config.")
except Exception as e:
    print("⚠️ Could not delete endpoint config:", str(e))