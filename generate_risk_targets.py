import boto3
import pandas as pd
import json
import time
import os

# Config
NUM_ROWS = 1000   # Test 10 vs 1000 for the whole xslx
INPUT_FILE = "data_files/credit_risk_with_descriptions_cleaned.csv"
OUTPUT_FILE = f"data_files/credit_risk_with_targets_{NUM_ROWS}.csv"
REGION = "us-west-2"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
MAX_TOKENS = 10  # keep small
TEMPERATURE = 0.0
SAVE_EVERY = 20

# Client
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

# Load original file
df = pd.read_csv(INPUT_FILE)
descriptions = df["description"].tolist()
total = len(descriptions)

# Resume if possible
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    targets = existing["target"].tolist()
    start = len(targets)
    print(f"🔄 Resuming from row {start}")
else:
    targets = []
    start = 0
    print("🆕 Starting from scratch")

start_time = time.time()

# Main loop
for i in range(start, total):
    prompt = build_prompt(descriptions[i])
    try:
        label = classify_description(prompt)
        if label not in ["good risk", "bad risk"]:
            print(f"[{i+1}/{total}] ⚠️ Unexpected output: {label}")
            label = "ERROR"
        else:
            print(f"[{i+1}/{total}] ✅ {label}")
    except Exception as e:
        label = "ERROR"
        print(f"[{i+1}/{total}] ❌ ERROR: {e}")
    targets.append(label)

    # Save every 20 rows
    if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
        df_partial = df.iloc[:i+1].copy()
        df_partial["target"] = targets
        df_partial.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Saved {i+1} rows to: {OUTPUT_FILE}")

    # Estimate time
    elapsed = time.time() - start_time
    avg = elapsed / (i - start + 1)
    eta = avg * (total - (i + 1))
    print(f"⏳ ETA: {round(eta/60, 1)} min")

# Final log
end_time = time.time()
print(f"\n✅ Done. Final file: {OUTPUT_FILE}")
print(f"⚡ Total time: {round((end_time - start_time)/60, 2)} minutes")