from typing import Any

from picsellia_cv_engine.models.contexts import PicselliaProcessingContext
from picsellia_cv_engine.models.parameters import Parameters


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


def create_picsellia_processing_context(processing_parameters: dict[str, Any]):
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
