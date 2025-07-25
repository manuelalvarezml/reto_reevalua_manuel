import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
import boto3

# Step 1: Setup
region = "us-west-2"

# Create a boto3 session for the correct region
boto_session = boto3.Session(region_name=region)

# Create the SageMaker session using the boto3 session
session = sagemaker.Session(boto_session=boto_session)

# AWS Sagemaker Execution Role
role = "arn:aws:iam::784608183649:role/SageMakerExecutionRole"

bucket = "reto-reevalua-s3-bucket"
s3_train_path = f"s3://{bucket}/data/train/train_data_sagemaker.csv"
s3_test_path = f"s3://{bucket}/data/test/test_data_sagemaker.csv"
s3_output_path = f"s3://{bucket}/output"

# Step 2: Get XGBoost container URI
xgb_container = sagemaker.image_uris.retrieve("xgboost", region=region, version="1.5-1")

# Step 3: Create estimator
xgb_estimator = Estimator(
    image_uri=xgb_container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=s3_output_path,
    sagemaker_session=session,
)

# Step 4: Set hyperparameters
xgb_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2,
    eval_metric="auc"
)

# Step 5: Define input (target column is last, no header)
train_input = TrainingInput(
    s3_data=s3_train_path,
    content_type="csv"
)

validation_input = TrainingInput(
    s3_data=s3_test_path,
    content_type="csv"
)

# Step 6: Launch training job
xgb_estimator.fit({"train": train_input, "validation": validation_input})