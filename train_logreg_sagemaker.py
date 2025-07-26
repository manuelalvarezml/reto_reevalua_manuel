# train_logreg_sagemaker.py
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# Set up session and role
session = sagemaker.Session()
role = "arn:aws:iam::784608183649:role/SageMakerExecutionRole"
bucket = 'reto-reevalua-s3-bucket'

# S3 paths
train_path = f"s3://{bucket}/data/train/train_data_sagemaker.csv"
test_path = f"s3://{bucket}/data/test/test_data_sagemaker.csv"

# Input channels
train_input = TrainingInput(train_path, content_type="text/csv")
test_input = TrainingInput(test_path, content_type="text/csv")

# Estimator
sklearn_estimator = SKLearn(
    entry_point="train_logreg.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.0-1",
    base_job_name="logreg-eval-job",
    py_version="py3",
    hyperparameters={"max_iter": 1000},
)

# Launch training job with both train and test channels
sklearn_estimator.fit({
    "train": train_input,
    "test": test_input
})