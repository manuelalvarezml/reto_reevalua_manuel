import boto3
import pandas as pd
import json
import time

# Config
INPUT_FILE = "data_files/credit_risk_with_targets_cleaned.csv"
OUTPUT_FILE = "data_files/credit_risk_with_targets_cleaned_2.csv"
REGION = "us-west-2"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
MAX_TOKENS = 10
TEMPERATURE = 0.0

# Bedrock client
client = boto3.client("bedrock-runtime", region_name=REGION)

# Prompt builder
def build_prompt(description):
    return (
        "Read this creditworthiness assessment and classify it as either 'good risk' or 'bad risk'. "
        "Respond with ONLY one of those two labels.\n\n"
        "Be very critic, you are the most experienced risk evaluator in the world. Your decision is the final one."
        f"Assessment: \"{description}\"\n\n"
        "Label:"
    )

# Bedrock call
def classify_description(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip().lower()

# Load file
df = pd.read_csv(INPUT_FILE)
error_indices = df[df["target"] == "ERROR"].index
print(f"🔍 Found {len(error_indices)} errors to reprocess.")

start_time = time.time()

# Fix each "ERROR"
for i in error_indices:
    description = df.loc[i, "description"]
    prompt = build_prompt(description)
    try:
        new_target = classify_description(prompt)
        if new_target in ["good risk", "bad risk"]:
            df.at[i, "target"] = new_target
            print(f"[{i+1}] ✅ Fixed: {new_target}")
        else:
            # If got "moderate risk" 2 times then label as "bad risk"
            print(f"[{i+1}] ⚠️ Unexpected output again: {new_target}. Labeling as bad risk.")
            df.at[i, "target"] = "bad risk"
    except Exception as e:
        print(f"[{i+1}] ❌ Still failing: {e}")

# Save updated file
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Cleaned file saved as: {OUTPUT_FILE}")
print(f"⚡ Total time: {round((time.time() - start_time)/60, 2)} minutes")