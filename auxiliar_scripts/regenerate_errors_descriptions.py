import boto3
import pandas as pd
import json
import time

# Configuration
INPUT_CSV = "data_files/credit_risk_with_descriptions_1000.csv"
OUTPUT_CSV = "data_files/credit_risk_with_descriptions_cleaned.csv"
REGION = "us-west-2"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
MAX_TOKENS = 250

# AWS Bedrock client
client = boto3.client("bedrock-runtime", region_name=REGION)

# Rebuild prompt for a row
def build_prompt(row):
    return (
        "Write a short, fluent sentence that assesses the applicant's creditworthiness. "
        "Avoid starting with generic phrases like 'Based on the information'. "
        "Be clear, direct, and sound like a human credit analyst.\n\n"
        "Here are the applicant’s details:\n"
        f"- Age: {row['Age']}\n"
        f"- Sex: {row['Sex']}\n"
        f"- Job: {row['Job']}\n"
        f"- Housing: {row['Housing']}\n"
        f"- Saving accounts: {row['Saving accounts']}\n"
        f"- Checking account: {row['Checking account']}\n"
        f"- Credit amount: {row['Credit amount']}\n"
        f"- Duration: {row['Duration']}\n"
        f"- Purpose: {row['Purpose']}"
    )

# Bedrock call
def query_bedrock(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# Load and filter
df = pd.read_csv(INPUT_CSV)
error_indices = df[df["description"] == "ERROR"].index

print(f"🔍 Found {len(error_indices)} rows with 'ERROR' descriptions.")

# Fix each error
for i in error_indices:
    prompt = build_prompt(df.loc[i])
    try:
        fixed_desc = query_bedrock(prompt)
        print(f"[{i}] ✅ {fixed_desc}")
        df.at[i, "description"] = fixed_desc
    except Exception as e:
        print(f"[{i}] ❌ Failed again: {e}")

# Save output
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Cleaned file saved as: {OUTPUT_CSV}")