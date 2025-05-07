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


def retrieve_picsellia_processing_parameters(
    processing_parameters: dict[str, Any],
) -> type[Parameters]:
    """
    Dynamically create a Parameters subclass from a dictionary of processing parameters.

    Args:
        processing_parameters (dict[str, Any]): Dictionary containing parameter names and example values.

    Returns:
        Type[Parameters]: A dynamically built subclass of Parameters with attributes based on the input.
    """

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
    Create a PicselliaProcessingContext from raw processing parameters.

    Args:
        processing_parameters (dict[str, Any]): Dictionary of parameters used to build the processing context.

    Returns:
        PicselliaProcessingContext: Initialized processing context object.
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
    Create a PicselliaTrainingContext from provided static parameter classes.

    Args:
        hyperparameters_cls (type): Class used to define hyperparameters.
        augmentation_parameters_cls (type): Class used to define augmentation parameters.
        export_parameters_cls (type): Class used to define export parameters.
        api_token (str | None): Optional Picsellia API token.
        host (str | None): Optional API host.
        organization_id (str | None): Optional organization ID.
        experiment_id (str | None): Optional experiment ID.

    Returns:
        PicselliaTrainingContext: Fully initialized training context.
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
