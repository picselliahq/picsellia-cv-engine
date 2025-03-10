import json
import os

from picsellia import Datalake
from picsellia.types.enums import InferenceType

from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data.dataset.uploader.classification.coco_classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)
from picsellia_cv_engine.models.steps.data.dataset.uploader.common.data_uploader import (
    DataUploader,
)
from picsellia_cv_engine.models.steps.data.dataset.uploader.object_detection.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)
from picsellia_cv_engine.models.steps.data.dataset.uploader.segmentation.segmentation_dataset_context_uploader import (
    SegmentationDatasetContextUploader,
)


def get_datalake_and_tag(
    context: PicselliaProcessingContext | None,
    datalake: Datalake | None,
    data_tag: str | None,
):
    """Retrieve datalake and data_tag from context or arguments."""
    if context:
        datalake = context.client.get_datalake(
            name=context.processing_parameters.datalake
        )
        data_tag = context.processing_parameters.data_tag
    if not datalake or not data_tag:
        raise ValueError("datalake and data_tag must not be None")
    return datalake, data_tag


def initialize_coco_data(dataset_context: CocoDatasetContext):
    """Ensure COCO data is initialized properly."""
    if dataset_context.coco_data and not dataset_context.coco_file_path:
        dataset_context.annotations_dir = (
            dataset_context.annotations_dir or "temp_annotations"
        )
        os.makedirs(dataset_context.annotations_dir, exist_ok=True)
        dataset_context.coco_file_path = os.path.join(
            dataset_context.annotations_dir, "annotations.json"
        )
        with open(dataset_context.coco_file_path, "w") as f:
            json.dump(dataset_context.coco_data, f)

    if dataset_context.coco_file_path and not dataset_context.coco_data:
        dataset_context.coco_data = dataset_context.load_coco_file_data()
    return dataset_context


def determine_inference_type(dataset_context: CocoDatasetContext, annotations: list):
    """Determine and set the inference type based on annotations."""
    first_annotation = annotations[0]
    if "segmentation" in first_annotation and first_annotation["segmentation"]:
        dataset_context.dataset_version.set_type(InferenceType.SEGMENTATION)
    elif "bbox" in first_annotation and first_annotation["bbox"]:
        dataset_context.dataset_version.set_type(InferenceType.OBJECT_DETECTION)
    elif "category_id" in first_annotation:
        dataset_context.dataset_version.set_type(InferenceType.CLASSIFICATION)
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_context.dataset_version.type}"
        )


def configure_dataset_type(dataset_context: CocoDatasetContext, annotations):
    """Configure dataset type if not already set."""
    if dataset_context.dataset_version.type == InferenceType.NOT_CONFIGURED:
        determine_inference_type(dataset_context, annotations)


def upload_images(
    dataset_context: CocoDatasetContext, datalake: Datalake, data_tag: str
):
    """Upload images to the dataset."""
    uploader = DataUploader(dataset_version=dataset_context.dataset_version)
    image_paths = [
        os.path.join(dataset_context.images_dir, img)
        for img in os.listdir(dataset_context.images_dir)
    ]
    uploader._add_images_to_dataset_version_in_batches(
        datalake=datalake, images_to_upload=image_paths, data_tags=[data_tag]
    )


def upload_dataset_context_based_on_type(
    dataset_context: CocoDatasetContext,
    datalake: Datalake,
    data_tag: str,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
):
    """
    Upload dataset context based on inference type.

    Supports Classification, Object Detection, and Segmentation inference types.
    """
    data_tags: list[str] = [data_tag]

    if dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetContextUploader(
            dataset_context=dataset_context,
        )
        classification_uploader.upload_dataset_context(
            datalake=datalake, data_tags=data_tags
        )

    elif dataset_context.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetContextUploader(
            dataset_context=dataset_context,
        )
        object_detection_uploader.upload_dataset_context(
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )

    elif dataset_context.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetContextUploader(
            dataset_context=dataset_context,
        )
        segmentation_uploader.upload_dataset_context(
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )


def upload_annotations_based_on_inference_type(
    dataset_context: CocoDatasetContext,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    """
    Upload annotations based on inference type.

    Supports Classification, Object Detection, and Segmentation inference types.
    """
    if dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetContextUploader(
            dataset_context=dataset_context,
        )
        classification_uploader.upload_annotations()

    elif dataset_context.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetContextUploader(
            dataset_context=dataset_context,
        )
        object_detection_uploader.upload_annotations(
            use_id=use_id, fail_on_asset_not_found=fail_on_asset_not_found
        )

    elif dataset_context.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetContextUploader(
            dataset_context=dataset_context,
        )
        segmentation_uploader.upload_annotations(
            use_id=use_id, fail_on_asset_not_found=fail_on_asset_not_found
        )
