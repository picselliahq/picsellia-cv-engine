import logging
from typing import Any

from picsellia.types.enums import InferenceType

from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.data.dataset.yolo_dataset_context import (
    YoloDatasetContext,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.classification.coco_classification_dataset_context_validator import (
    CocoClassificationDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.common.dataset_context_validator import (
    DatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.object_detection.coco_object_detection_dataset_context_validator import (
    CocoObjectDetectionDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.object_detection.yolo_object_detection_dataset_context_validator import (
    YoloObjectDetectionDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.segmentation.coco_segmentation_dataset_context_validator import (
    CocoSegmentationDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.segmentation.yolo_segmentation_dataset_context_validator import (
    YoloSegmentationDatasetContextValidator,
)

logger = logging.getLogger(__name__)


def get_validator_for_dataset_context(
    dataset_context: TBaseDatasetContext, fix_annotation: bool = True
) -> Any:
    """Retrieves the appropriate validator for a given dataset context.

    Args:
        dataset_context (TBaseDatasetContext): The dataset context to validate.
        fix_annotation (bool, optional): A flag to indicate whether to automatically fix errors (default is True).

    Returns:
        Any: The validator instance or None if the dataset type is unsupported.
    """
    validators = {
        (
            CocoDatasetContext,
            InferenceType.CLASSIFICATION,
        ): CocoClassificationDatasetContextValidator,
        (
            CocoDatasetContext,
            InferenceType.OBJECT_DETECTION,
        ): CocoObjectDetectionDatasetContextValidator,
        (
            CocoDatasetContext,
            InferenceType.SEGMENTATION,
        ): CocoSegmentationDatasetContextValidator,
        (
            YoloDatasetContext,
            InferenceType.OBJECT_DETECTION,
        ): YoloObjectDetectionDatasetContextValidator,
        (
            YoloDatasetContext,
            InferenceType.SEGMENTATION,
        ): YoloSegmentationDatasetContextValidator,
    }

    inference_type = dataset_context.dataset_version.type

    if inference_type == InferenceType.NOT_CONFIGURED:
        return DatasetContextValidator(dataset_context=dataset_context)

    validator_class = validators.get((type(dataset_context), inference_type))

    if validator_class is None:
        logger.warning(
            f"Dataset type '{type(dataset_context).__name__}' with inference type '{inference_type.name}' is not supported. Skipping validation."
        )
        return None

    return validator_class(
        dataset_context=dataset_context, fix_annotation=fix_annotation
    )
