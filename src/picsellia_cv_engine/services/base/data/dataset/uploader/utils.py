import json
import os

from picsellia import Datalake
from picsellia.types.enums import InferenceType

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.data import (
    CocoDataset,
)
from picsellia_cv_engine.services.base.data.dataset.uploader.classification import (
    ClassificationDatasetUploader,
)
from picsellia_cv_engine.services.base.data.dataset.uploader.common import (
    DatasetUploader,
)
from picsellia_cv_engine.services.base.data.dataset.uploader.object_detection import (
    ObjectDetectionDatasetUploader,
)
from picsellia_cv_engine.services.base.data.dataset.uploader.segmentation import (
    SegmentationDatasetUploader,
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


def initialize_coco_data(dataset: CocoDataset):
    """Ensure COCO data is initialized properly."""
    if dataset.coco_data and not dataset.coco_file_path:
        dataset.annotations_dir = dataset.annotations_dir or "temp_annotations"
        os.makedirs(dataset.annotations_dir, exist_ok=True)
        dataset.coco_file_path = os.path.join(
            dataset.annotations_dir, "annotations.json"
        )
        with open(dataset.coco_file_path, "w") as f:
            json.dump(dataset.coco_data, f)

    if dataset.coco_file_path and not dataset.coco_data:
        dataset.coco_data = dataset.load_coco_file_data()
    return dataset


def determine_inference_type(dataset: CocoDataset, annotations: list):
    """Determine and set the inference type based on annotations."""
    first_annotation = annotations[0]
    if "segmentation" in first_annotation and first_annotation["segmentation"]:
        dataset.dataset_version.set_type(InferenceType.SEGMENTATION)
    elif "bbox" in first_annotation and first_annotation["bbox"]:
        dataset.dataset_version.set_type(InferenceType.OBJECT_DETECTION)
    elif "category_id" in first_annotation:
        dataset.dataset_version.set_type(InferenceType.CLASSIFICATION)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset.dataset_version.type}")


def configure_dataset_type(dataset: CocoDataset, annotations):
    """Configure dataset type if not already set."""
    if dataset.dataset_version.type == InferenceType.NOT_CONFIGURED:
        determine_inference_type(dataset, annotations)


def upload_images(dataset: CocoDataset, datalake: Datalake, data_tag: str):
    """Upload images to the dataset."""
    uploader = DatasetUploader(dataset_version=dataset.dataset_version)
    image_paths = [
        os.path.join(dataset.images_dir, img) for img in os.listdir(dataset.images_dir)
    ]
    uploader._add_images_to_dataset_version_in_batches(
        datalake=datalake, images_to_upload=image_paths, data_tags=[data_tag]
    )


def upload_dataset_based_on_type(
    dataset: CocoDataset,
    datalake: Datalake,
    data_tag: str,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
    replace_annotations: bool = False,
):
    """
    Upload dataset based on inference type.

    Supports Classification, Object Detection, and Segmentation inference types.
    """
    data_tags: list[str] = [data_tag]

    if dataset.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetUploader(
            dataset=dataset,
        )
        classification_uploader.upload_dataset(datalake=datalake, data_tags=data_tags)

    elif dataset.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetUploader(
            dataset=dataset,
        )
        object_detection_uploader.upload_dataset(
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
            replace_annotations=replace_annotations,
        )

    elif dataset.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetUploader(
            dataset=dataset,
        )
        segmentation_uploader.upload_dataset(
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
            replace_annotations=replace_annotations,
        )


def upload_annotations_based_on_inference_type(
    dataset: CocoDataset,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
    replace_annotations: bool = False,
) -> None:
    """
    Upload annotations based on inference type.

    Supports Classification, Object Detection, and Segmentation inference types.
    """
    if dataset.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetUploader(
            dataset=dataset,
        )
        classification_uploader.upload_annotations()

    elif dataset.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetUploader(
            dataset=dataset,
        )
        object_detection_uploader.upload_annotations(
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
            replace_annotations=replace_annotations,
        )

    elif dataset.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetUploader(
            dataset=dataset,
        )
        segmentation_uploader.upload_annotations(
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
            replace_annotations=replace_annotations,
        )
