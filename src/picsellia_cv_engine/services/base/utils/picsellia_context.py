from typing import Any

from picsellia_cv_engine.core.contexts import (
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
    HyperParameters,
    Parameters,
)


def retrieve_picsellia_processing_parameters(processing_parameters: dict[str, Any]):
    class ProcessingParameters(Parameters):
        def __init__(self, log_data):
            super().__init__(log_data)
            for key, value in processing_parameters.items():
                expected_type = type(value)
                setattr(
                    self,
                    key,
                    self.extract_parameter(keys=[key], expected_type=expected_type),
                )

    return ProcessingParameters


def create_picsellia_processing_context(
    processing_parameters: dict[str, Any],
) -> PicselliaProcessingContext:
    """
    Create a Picsellia processing context.

    Returns:
        PicselliaProcessingContext: Picsellia processing context object.
    """
    processing_parameters_cls = retrieve_picsellia_processing_parameters(
        processing_parameters
    )
    context = PicselliaProcessingContext(
        processing_parameters_cls=processing_parameters_cls,
    )
    return context


def create_picsellia_training_context(
    hyperparameters_cls: type[HyperParameters],
    augmentation_parameters_cls: type[AugmentationParameters],
    export_parameters_cls: type[ExportParameters],
    api_token: str | None = None,
    host: str | None = None,
    organization_id: str | None = None,
    experiment_id: str | None = None,
) -> PicselliaTrainingContext:
    """
    Create a Picsellia training context using static parameter classes.

    Args:
        hyperparameters_cls: Class used to extract hyperparameters
        augmentation_parameters_cls: Class used to extract augmentation parameters
        export_parameters_cls: Class used to extract export parameters
        api_token: API token for authentication
        host: Host URL for the Picsellia API
        organization_id: Organization ID for the Picsellia account
        experiment_id: Experiment ID for the training run

    Returns:
        A fully initialized PicselliaTrainingContext.
    """
    return PicselliaTrainingContext(
        hyperparameters_cls=hyperparameters_cls,
        augmentation_parameters_cls=augmentation_parameters_cls,
        export_parameters_cls=export_parameters_cls,
        api_token=api_token,
        host=host,
        organization_id=organization_id,
        experiment_id=experiment_id,
    )
