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
from picsellia_cv_engine.core.parameters.base_parameters import TParameters


def create_local_processing_context(
    processing_parameters_cls: type[TParameters],
    api_token: str,
    organization_name: str,
    job_type: ProcessingType,
    input_dataset_version_id: str,
    output_dataset_version_name: str | None = None,
    model_version_id: str | None = None,
    processing_parameters: dict[str, Any] | None = None,
    working_dir: str | None = None,
    host: str | None = None,
) -> LocalProcessingContext:
    """
    Create a local processing context for running a processing pipeline outside of Picsellia.

    This is typically used for development and testing, with full local control over input/output paths
    and parameter overrides.

    Args:
        processing_parameters_cls (type[TParameters]): A subclass of `Parameters` used to define typed inputs.
        api_token (str): API token for authentication with Picsellia.
        organization_name (str): Name of the Picsellia organization.
        job_type (ProcessingType): Type of processing job (e.g., `PRE_ANNOTATION`, `DATASET_VERSION_CREATION`).
        input_dataset_version_id (str): ID of the dataset version used as input.
        output_dataset_version_name (str | None): Optional name for the output dataset version.
        model_version_id (str | None): Optional ID of a model version to include in the context.
        processing_parameters (dict[str, Any] | None): Raw values to override defaults in the processing parameters.
        working_dir (str | None): Optional working directory for local file operations.
        host (str | None): Optional Picsellia API host override.

    Returns:
        LocalProcessingContext[TParameters]: A fully initialized local processing context.
    """
    context = LocalProcessingContext(
        processing_parameters_cls=processing_parameters_cls,
        processing_parameters=processing_parameters,
        api_token=api_token,
        organization_name=organization_name,
        host=host,
        job_type=job_type,
        input_dataset_version_id=input_dataset_version_id,
        output_dataset_version_name=output_dataset_version_name,
        model_version_id=model_version_id,
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
    working_dir: str | None = None,
    host: str | None = None,
) -> LocalTrainingContext:
    """
    Create a local training context for testing model training logic locally.

    This context allows for local execution of training steps with parameters pulled from experiment logs.

    Args:
        hyperparameters_cls (type): Class defining training hyperparameters.
        augmentation_parameters_cls (type): Class defining augmentation strategy parameters.
        export_parameters_cls (type): Class defining model export configuration.
        api_token (str): API token to authenticate with Picsellia.
        organization_name (str): Name of the organization linked to the experiment.
        experiment_id (str): Experiment ID from which parameter logs are retrieved.
        working_dir (str | None): Optional local working directory.
        host (str | None): Optional Picsellia host override.

    Returns:
        LocalTrainingContext: Fully initialized context for local model training.
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
