import boto3
client = boto3.client("bedrock", region_name="us-west-2")
response = client.list_foundation_models()
for m in response["modelSummaries"]:
    if "anthropic" in m["modelId"]:
        print(m["modelId"])