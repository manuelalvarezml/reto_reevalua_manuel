import boto3
import pandas as pd
import json
import time

# Configuration variables
NUM_ROWS = 1000   # Test 10 vs 1000 for the whole xslx

INPUT_FILE = "data_files/credir_risk_reto.xlsx"
OUTPUT_CSV = f"data_files/credit_risk_with_descriptions_{NUM_ROWS}.csv"

REGION = "us-west-2"
# MODEL_ID = "anthropic.claude-instant-v1"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0" # Cheaper, faster and better for structured generation than instant

MAX_TOKENS = 250
# SLEEP_TIME = 0.7  # (100 RPM as an initial value)
SLEEP_TIME = 0  # (incremented to 1M RPM)

# AWS Bedrock Client
client = boto3.client("bedrock-runtime", region_name=REGION)

# Prompt generator
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

# Bedrock Call
def query_bedrock(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": prompt}
        ],
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

# Processing
df = pd.read_excel(INPUT_FILE)

if NUM_ROWS < len(df):
    df = df.head(NUM_ROWS)

descriptions = []
df_len = len(df)

start_time = time.time()
checkpoints = [20 * i for i in range(1, (df_len // 20) + 2)]

# Load saved progress if file exists
try:
    existing_df = pd.read_csv(OUTPUT_CSV)
    existing_descriptions = existing_df["description"].tolist()
    start_index = len(existing_descriptions)
    print(f"🔄 Resuming from row {start_index}")
except FileNotFoundError:
    existing_descriptions = []
    start_index = 0
    print("🔄 No existing CSV found, starting from scratch.")

# Main loop

for i in range(start_index, df_len):
    row = df.iloc[i]
    prompt = build_prompt(row)
    try:
        desc = query_bedrock(prompt)
        print(f"[{i+1}/{df_len}] ✅ {desc}")
    except Exception as e:
        desc = "ERROR"
        print(f"[{i+1}/{df_len}] ❌ ERROR: {e}")
    existing_descriptions.append(desc)

    # Save progress every 20 rows
    if (i + 1) % 20 == 0 or (i + 1) == df_len:
        df_partial = df.head(i + 1).copy()
        df_partial["description"] = existing_descriptions
        df_partial.to_csv(OUTPUT_CSV, index=False)
        print(f"💾 Autosaved {i+1} rows to: {OUTPUT_CSV}")

    # Estimate remaining time
    elapsed = time.time() - start_time
    avg_time = elapsed / (i - start_index + 1)
    est_remaining = avg_time * (df_len - (i + 1))
    print(f"⏳ Estimated time remaining: {round(est_remaining / 60, 1)} minutes")

# Final save
df["description"] = descriptions
df.to_csv(OUTPUT_CSV, index=False)

end_time = time.time()
print(f"\n✅ Saved final file as: {OUTPUT_CSV}")
print(f"⚡ Total time: {round((end_time - start_time)/60, 2)} minutes")