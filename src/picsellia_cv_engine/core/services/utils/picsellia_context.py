from picsellia_cv_engine.core.contexts import (
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
    HyperParameters,
)
from picsellia_cv_engine.core.parameters.base_parameters import TParameters


def create_picsellia_processing_context(
    processing_parameters_cls: type[TParameters],
) -> PicselliaProcessingContext:
    """
    Create a remote PicselliaProcessingContext using a static class to define parameters.

    This context is used during pipeline execution on the Picsellia platform.

    Args:
        processing_parameters_cls (type[TParameters]): A class inheriting from `Parameters` defining expected processing parameters.

    Returns:
        PicselliaProcessingContext: An initialized context for use in remote processing pipelines.
    """
    context = PicselliaProcessingContext(
        processing_parameters_cls=processing_parameters_cls,
    )
    return context


def create_picsellia_training_context(
    hyperparameters_cls: type[HyperParameters],
    augmentation_parameters_cls: type[AugmentationParameters],
    export_parameters_cls: type[ExportParameters],
) -> PicselliaTrainingContext:
    """
    Create a remote PicselliaTrainingContext using static parameter classes.

    This context is used during model training executed on the Picsellia platform.

    Args:
        hyperparameters_cls (type): Class defining hyperparameters (inherits from `HyperParameters`).
        augmentation_parameters_cls (type): Class defining augmentation parameters.
        export_parameters_cls (type): Class defining export/export format parameters.

    Returns:
        PicselliaTrainingContext: An initialized context for remote training pipelines.
    """
    return PicselliaTrainingContext(
        hyperparameters_cls=hyperparameters_cls,
        augmentation_parameters_cls=augmentation_parameters_cls,
        export_parameters_cls=export_parameters_cls,
    )
