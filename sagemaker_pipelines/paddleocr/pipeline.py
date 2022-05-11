"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

from botocore.exceptions import ClientError
import boto3

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )



def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="PaddleOCRPackageGroup",
    pipeline_name="PaddleOCRPipelne",
    base_job_prefix="PaddleOCR",
    project_id="SageMakerProjectId"
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if not default_bucket: 
        default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    # parametersagemaker_sessions for pipeline execution
    sess = boto3.Session()
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.p2.xlarge"
    )
    inference_instance_type = ParameterString(
        name="InferenceInstanceType", default_value="ml.p2.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://{}/PaddleOCR/input/data".format(default_bucket)
    )
    account = sess.client("sts").get_caller_identity()["Account"]
    region = sess.region_name
    data_generate_image_name = "generate-ocr-train-data"
    train_image_name = "paddle"
    data_generate_image = "{}.dkr.ecr.{}.amazonaws.com/{}".format(account, region, data_generate_image_name)
    
    script_processor = ScriptProcessor(
        image_uri=data_generate_image,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/paddle-ocr-data-generation",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="GenerateOCRTrainingData",
        processor=script_processor,
        outputs=[
            ProcessingOutput(output_name="data", source="/opt/ml/processing/input/data"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/PaddleOCRTrain"


    image = "{}.dkr.ecr.{}.amazonaws.com/{}".format(account, region, train_image_name)

    training_image_uri = image 
    hyperparameters = {"epoch_num": 10,
                  "print_batch_step":5,
                  "save_epoch_step":30,
                  'pretrained_model':'/opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train/best_accuracy'}

    paddle_train = Estimator(
        image_uri=training_image_uri,
        instance_type=training_instance_type,
        role=role,
        instance_count = 1,
        output_path=model_path,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/paddleocr-train",
        hyperparameters=hyperparameters,
#         acc: 0.2007992007992008, norm_edit_dis: 0.7116550116550118, fps: 97.10778964378831, best_epoch: 9
        metric_definitions=[
               {'Name': 'validation:acc', 'Regex': '.*best metric,.*acc:(.*?),'},
               {'Name': 'validation:norm_edit_dis', 'Regex': '.*best metric,.*norm_edit_dis:(.*?),'}
        ]

    )


    step_train = TrainingStep(
        name="TrainPaddleOCRModel",
        estimator=paddle_train,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "data"
                ].S3Output.S3Uri,
                content_type="image/jpeg")
        }
    )




    inference_image_uri = image
    step_register = RegisterModel(
        name="RegisterPaddleOCRModel",
        estimator=paddle_train,
        image_uri=inference_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.p2.xlarge"],
        transform_instances=["ml.p2.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status
    )
    
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=step_train.properties.FinalMetricDataList[0].Value,
        right=0.8,  # You can change the threshold here
    )
    
    step_cond = ConditionStep(
        name="PaddleOCRAccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )


    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps = [step_process, step_train, step_cond],
#         steps=[step_train, step_register],
        sagemaker_session=sagemaker_session,
    )
    return pipeline