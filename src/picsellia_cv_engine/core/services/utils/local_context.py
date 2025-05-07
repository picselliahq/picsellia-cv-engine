import dataclasses
from typing import Any

from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
)
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
    HyperParameters,
)


def infer_type(value: str) -> Any:
    """
    Infer the Python type of string value.

    Tries to cast the input to int, then float, and falls back to string.

    Args:
        value (str): The string to convert.

    Returns:
        Any: The value cast to int, float, or left as string.
    """
    try:
        return int(value)  # Check if it's an integer
    except ValueError:
        try:
            return float(value)  # Check if it's a float
        except ValueError:
            return value  # Default to string


def create_local_processing_parameters(processing_parameters: dict[str, Any]) -> Any:
    """
    Dynamically create a local ProcessingParameters dataclass from a dictionary.

    Args:
        processing_parameters (dict[str, Any]): Dictionary of parameter values.

    Returns:
        ProcessingParameters: A dataclass instance with inferred types and values.
    """

    @dataclasses.dataclass
    class ProcessingParameters:
        defaulted_keys: set[str] = dataclasses.field(default_factory=set)

        def __init__(self):
            # Dynamically add attributes and infer types
            for key, value in processing_parameters.items():
                inferred_value = infer_type(value)
                setattr(self, key, inferred_value)

            # Initialize empty defaulted keys for logging purposes
            self.defaulted_keys = set()

        def to_dict(self) -> dict[str, Any]:
            """Convert the parameters to a dictionary."""
            filtered_dict = {
                key: value
                for key, value in self.__dict__.items()
                if key not in ["defaulted_keys"]
            }
            return dict(sorted(filtered_dict.items()))

        def set_defaulted_key(self, key: str):
            """Log a key as defaulted."""
            self.defaulted_keys.add(key)

    return ProcessingParameters()


def create_local_processing_context(
    api_token: str,
    organization_name: str,
    job_type: ProcessingType,
    input_dataset_version_id: str,
    processing_parameters: dict[str, Any],
    output_dataset_version_name: str | None = None,
    model_version_id: str | None = None,
    working_dir: str | None = None,
) -> LocalProcessingContext:
    """
    Create a local Picsellia processing context for testing a processing pipeline.

    Args:
        api_token (str): API token for authentication.
        organization_name (str): Name of the organization.
        job_type (ProcessingType): Type of processing (e.g., PRE_ANNOTATION).
        input_dataset_version_id (str): ID of the dataset version to use as input.
        processing_parameters (dict[str, Any]): Parameters for the processing step.
        output_dataset_version_name (str | None): Name of the output dataset version (optional).
        model_version_id (str | None): ID of the model version to use (optional).
        working_dir (str | None): Working directory to store files locally (optional).

    Returns:
        LocalProcessingContext: Initialized processing context.
    """
    processing_parameters_data = create_local_processing_parameters(
        processing_parameters
    )
    context = LocalProcessingContext(
        api_token=api_token,
        organization_name=organization_name,
        job_type=job_type,
        input_dataset_version_id=input_dataset_version_id,
        output_dataset_version_name=output_dataset_version_name,
        model_version_id=model_version_id,
        processing_parameters=processing_parameters_data,
        working_dir=working_dir,
    )
    return context


def create_local_training_context(
    hyperparameters_cls: type[HyperParameters],
    augmentation_parameters_cls: type[AugmentationParameters],
    export_parameters_cls: type[ExportParameters],
    api_token: str,
    organization_name: str,
    experiment_id: str,
    host: str | None = None,
    working_dir: str | None = None,
) -> LocalTrainingContext:
    """
    Create a local training context for testing training workflows.

    Args:
        hyperparameters_cls (type): Class used to extract hyperparameters.
        augmentation_parameters_cls (type): Class used to extract augmentation parameters.
        export_parameters_cls (type): Class used to extract export parameters.
        api_token (str): API token for Picsellia access.
        organization_name (str): Name of the organization.
        experiment_id (str): ID of the experiment to attach logs/artifacts.
        host (str | None): Picsellia API host (optional).
        working_dir (str | None): Local working directory (optional).

    Returns:
        LocalTrainingContext: Initialized training context.
    """
    return LocalTrainingContext(
        api_token=api_token,
        organization_name=organization_name,
        experiment_id=experiment_id,
        host=host,
        hyperparameters_cls=hyperparameters_cls,
        augmentation_parameters_cls=augmentation_parameters_cls,
        export_parameters_cls=export_parameters_cls,
        working_dir=working_dir,
    )
