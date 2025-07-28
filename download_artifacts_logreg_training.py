import boto3
import os
import tarfile

# Update these
bucket = "sagemaker-us-west-2-784608183649"
job_name = "logreg-eval-job-2025-07-26-20-08-25-636"
artifact_path = f"{job_name}/output/model.tar.gz"
download_path = "downloaded_artifacts"
local_tar_path = os.path.join(download_path, "model.tar.gz")

# Create folder
os.makedirs(download_path, exist_ok=True)

# Download from S3
s3 = boto3.client("s3")
s3.download_file(bucket, artifact_path, local_tar_path)

# Extract
with tarfile.open(local_tar_path, "r:gz") as tar:
    tar.extractall(path=download_path)

print("✅ Download complete. Files extracted:")
print(f"- model.joblib")
print(f"- evaluation.txt")
if os.path.exists(os.path.join(download_path, "predictions.csv")):
    print("- predictions.csv")