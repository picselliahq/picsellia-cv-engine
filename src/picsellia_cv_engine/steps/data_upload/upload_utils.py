import json
import os

from picsellia import Datalake
from picsellia.types.enums import InferenceType

from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data_upload.data_uploader import DataUploader


def get_datalake_and_tag(
    context: PicselliaProcessingContext | None,
    datalake: Datalake | None,
    data_tag: str | None,
):
    """Retrieve datalake and data_tag from context or arguments."""
    if context:
        datalake = context.client.get_datalake(context.processing_parameters.datalake)
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
