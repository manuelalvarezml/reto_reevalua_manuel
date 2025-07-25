import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

# model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0" # Quota: 3/min
model_id = "anthropic.claude-instant-v1" # Quota: 100/min

# Claude 3.5 requiere formato 'messages' + 'anthropic_version'
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {
            "role": "user",
            "content": "Describe the credit risk of a 35-year-old skilled worker applying for a 24-month loan with a low credit amount."
        }
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

result = json.loads(response["body"].read())
print("✅ Respuesta:")
print(result["content"][0]["text"])