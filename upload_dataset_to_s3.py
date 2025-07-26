import boto3
import os

bucket_name = "reto-reevalua-s3-bucket"
local_root = "dataset_for_sagemaker"
s3_root = "data"  # corresponds to s3://reto-reevalua-s3-bucket/data/

s3 = boto3.client("s3")

# Walk through all subdirectories and files
for root, _, files in os.walk(local_root):
    for file in files:
        if file == ".DS_Store":
            continue  # Skip macOS metadata
        local_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_path, local_root)
        s3_key = f"{s3_root}/{relative_path}"

        print(f"⏫ Uploading {local_path} → s3://{bucket_name}/{s3_key}")
        s3.upload_file(local_path, bucket_name, s3_key)

print("✅ All files uploaded and overridden if they existed.")