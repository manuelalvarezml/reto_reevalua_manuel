import boto3
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
NUM_ROWS = 10  # Full dataset
INPUT_FILE = "credir_risk_reto.xlsx"
OUTPUT_CSV = f"credit_risk_with_descriptions_{NUM_ROWS}.csv"
REGION = "us-west-2"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
MAX_TOKENS = 250
MAX_WORKERS = 5  # Tune this based on your bandwidth and Bedrock reliability

# Bedrock client
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

# Bedrock call
def query_bedrock(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"ERROR: {e}"

# Load data
df = pd.read_excel(INPUT_FILE)
if NUM_ROWS < len(df):
    df = df.head(NUM_ROWS)

# Process in parallel
start_time = time.time()
prompts = [build_prompt(row) for _, row in df.iterrows()]

error_count = 0
descriptions = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(query_bedrock, prompt) for prompt in prompts]

    for i, future in enumerate(as_completed(futures), start=1):
        result = future.result()
        descriptions.append(result)

        if result.startswith("ERROR:"):
            error_count += 1

        if i % 10 == 0:
            print(f"⏳ Processed {i}/{len(prompts)} rows — sleeping 20s to avoid throttling")
            # time.sleep(30)
            print(f"❌ Total errors: {error_count}")
        elif i % 10 == 0 or i == len(prompts):
            print(f"✅ Processed {i}/{len(prompts)} rows")
            print(f"❌ Total errors: {error_count}")

end_time = time.time()

# Output
df["description"] = descriptions
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved file as: {OUTPUT_CSV}")
print(f"⚡ Completed in {round(end_time - start_time, 2)} seconds.")