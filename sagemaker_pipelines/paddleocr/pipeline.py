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
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
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

def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """ Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    #Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None

def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

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

    # parameters for pipeline execution
#     processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
#     processing_instance_type = ParameterString(
#         name="ProcessingInstanceType", default_value="ml.m5.xlarge"
#     )
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
        default_value="s3://{}/DEMO-paddle-byo/".format(default_bucket)
    )
    
    training_image_name = "paddle"
    inference_image_name = "paddle"

    # processing step for feature engineering
#     try:
#         processing_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=processing_image_name)['ContainerImage']
#     except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
#         processing_image_uri = sagemaker.image_uris.retrieve(
#             framework="xgboost",
#             region=region,
#             version="1.0-1",
#             py_version="py3",
#             instance_type=processing_instance_type,
#         )
#     script_processor = ScriptProcessor(
#         image_uri=processing_image_uri,
#         instance_type=processing_instance_type,
#         instance_count=processing_instance_count,
#         base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
#         command=["python3"],
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )
#     step_process = ProcessingStep(
#         name="PreprocessAbaloneData",
#         processor=script_processor,
#         outputs=[
#             ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
#             ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
#             ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
#         ],
#         code=os.path.join(BASE_DIR, "preprocess.py"),
#         job_arguments=["--input-data", input_data],
#     )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/PaddleOCRTrain"

#     try:
#         print(training_image_name)
#         training_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=training_image_name)['ContainerImage']
#     except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
#         training_image_uri = sagemaker.image_uris.retrieve(
#             framework="xgboost",
#             region=region,
#             version="1.0-1",
#             py_version="py3",
#             instance_type=training_instance_type,
#         )

    training_image_uri = "230755935769.dkr.ecr.us-east-1.amazonaws.com/paddle:latest"
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
    )


    step_train = TrainingStep(
        name="TrainPaddleOCRModel",
        estimator=paddle_train,
        inputs={
            "training": TrainingInput(
                s3_data=input_data,
                content_type="text/csv",
            )
        },
    )

    # processing step for evaluation
#     script_eval = ScriptProcessor(
#         image_uri=training_image_uri,
#         command=["python3"],
#         instance_type=processing_instance_type,
#         instance_count=1,
#         base_job_name=f"{base_job_prefix}/script-abalone-eval",
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )
#     evaluation_report = PropertyFile(
#         name="AbaloneEvaluationReport",
#         output_name="evaluation",
#         path="evaluation.json",
#     )
#     step_eval = ProcessingStep(
#         name="EvaluateAbaloneModel",
#         processor=script_eval,
#         inputs=[
#             ProcessingInput(
#                 source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#                 destination="/opt/ml/processing/model",
#             ),
#             ProcessingInput(
#                 source=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "test"
#                 ].S3Output.S3Uri,
#                 destination="/opt/ml/processing/test",
#             ),
#         ],
#         outputs=[
#             ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
#         ],
#         code=os.path.join(BASE_DIR, "evaluate.py"),
#         property_files=[evaluation_report],
#     )

#     # register model step that will be conditionally executed
#     model_metrics = ModelMetrics(
#         model_statistics=MetricsSource(
#             s3_uri="{}/evaluation.json".format(
#                 step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
#             ),
#             content_type="application/json"
#         )
#     )

#     try:
#         inference_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=inference_image_name)['ContainerImage']
#     except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
#         inference_image_uri = sagemaker.image_uris.retrieve(
#             framework="xgboost",
#             region=region,
#             version="1.0-1",
#             py_version="py3",
#             instance_type=inference_instance_type,
#         )

    inference_image_uri = "230755935769.dkr.ecr.us-east-1.amazonaws.com/paddle:latest"
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
        approval_status=model_approval_status,
#         model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
#     cond_lte = ConditionLessThanOrEqualTo(
#         left=JsonGet(
#             step_name=step_eval.name,
#             property_file=evaluation_report,
#             json_path="regression_metrics.mse.value"
#         ),
#         right=6.0,
#     )
#     step_cond = ConditionStep(
#         name="CheckMSEAbaloneEvaluation",
#         conditions=[cond_lte],
#         if_steps=[step_register],
#         else_steps=[],
#     )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
#             processing_instance_type,
#             processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_train, step_register],
        sagemaker_session=sagemaker_session,
    )
    return pipeline