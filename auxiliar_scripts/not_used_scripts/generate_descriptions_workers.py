import boto3
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from threading import Lock

# Configuration
NUM_ROWS = 100  # Test 10 vs 1000 for the whole xslx
INPUT_FILE = "credir_risk_reto.xlsx"
OUTPUT_CSV = f"credit_risk_with_descriptions_{NUM_ROWS}.csv"
REGION = "us-west-2"
# MODEL_ID = "anthropic.claude-instant-v1"
MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"  # Cheaper, faster and better for structured generation than instant
MAX_TOKENS = 250
MAX_WORKERS = 2  # Tune this based on your bandwidth and Bedrock reliability

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

# Counters for logging
completed_count = 0
error_count = 0
lock = Lock()

# Bedrock call
def query_bedrock(prompt, retries=3):
    for attempt in range(retries):
        try:
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
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = random.uniform(0.5, 1.5)
                print(f"Retrying after error: {e} (sleeping {sleep_time:.2f}s)")
                time.sleep(sleep_time)
            else:
                return f"ERROR: {e}"

# Load data
df = pd.read_excel(INPUT_FILE)
if NUM_ROWS < len(df):
    df = df.head(NUM_ROWS)

# Process in parallel
start_time = time.time()
prompts = [build_prompt(row) for _, row in df.iterrows()]
descriptions = []

def wrapped_query(prompt, idx):
    global completed_count, error_count
    result = query_bedrock(prompt)
    with lock:
        completed_count += 1
        if completed_count % 50 == 0 or completed_count == len(prompts):
            print(f"✅ Completed {completed_count}/{len(prompts)} prompts.")
        if result.startswith("ERROR:"):
            error_count += 1
    return result

futures = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for i, prompt in enumerate(prompts):
        futures.append(executor.submit(wrapped_query, prompt, i))

    for future in as_completed(futures):
        descriptions.append(future.result())
        time.sleep(random.uniform(0.3, 0.6))  # Throttle actual LLM request spacing

end_time = time.time()

# Output
df["description"] = descriptions
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved file as: {OUTPUT_CSV}")
print(f"⚡ Completed in {round(end_time - start_time, 2)} seconds.")
print(f"📈 Total rows processed: {completed_count}")
print(f"❌ Total errors: {error_count}")