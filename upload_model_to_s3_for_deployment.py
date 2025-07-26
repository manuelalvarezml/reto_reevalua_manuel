import boto3

s3 = boto3.client("s3")
bucket = "reto-reevalua-s3-bucket"
artifact_path = "deployed_model/model.tar.gz"

s3.upload_file("downloaded_artifacts/model.tar.gz", bucket, artifact_path)
print("✅ Uploaded model.tar.gz to S3")