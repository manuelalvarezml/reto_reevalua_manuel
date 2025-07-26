import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ---- Config ----
bucket = "reto-reevalua-s3-bucket"
model_path = "deployed_model/model.tar.gz"
role = "arn:aws:iam::784608183649:role/SageMakerExecutionRole"
model_s3_uri = f"s3://{bucket}/{model_path}"
endpoint_name = "logreg-fraud-detector-manuel"

# ---- Session ----
sagemaker_session = sagemaker.Session()

# ---- Model ----
model = SKLearnModel(
    model_data=model_s3_uri,
    role=role,
    entry_point="train_logreg.py",  # This script should include inference code
    framework_version="0.23-1",
    sagemaker_session=sagemaker_session
)

# ---- Deploy ----
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print("✅ Model deployed!")
print(f"🛰️  Endpoint name: {endpoint_name}")