from loguru import logger

try:
    import boto3
    import sagemaker
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import get_huggingface_llm_image_uri
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.model.utils import ResourceManager
from llm_engineering.settings import settings

from .config import hugging_face_deploy_config, model_resource_config
from .sagemaker_huggingface import (DeploymentService,
                                    SagemakerHuggingfaceStrategy)


def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."
    assert settings.AWS_REGION, "AWS_REGION is not set."
    assert settings.AWS_ACCESS_KEY, "AWS_ACCESS_KEY is not set."
    assert settings.AWS_SECRET_KEY, "AWS_SECRET_KEY is not set."

    logger.info(f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {settings.HF_MODEL_ID}")

    # Create a boto3 session with the specified credentials
    boto_session = boto3.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
    region_name=settings.AWS_REGION
    )

    # Create the SageMaker session
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0", session=sagemaker_session)

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)



    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
