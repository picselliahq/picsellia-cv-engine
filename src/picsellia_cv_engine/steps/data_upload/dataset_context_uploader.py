from picsellia import Datalake
from picsellia.types.enums import InferenceType

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data_upload.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)
from picsellia_cv_engine.models.steps.data_upload.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)
from picsellia_cv_engine.models.steps.data_upload.segmentation_dataset_context_uploader import (
    SegmentationDatasetContextUploader,
)
from picsellia_cv_engine.steps.data_upload.upload_utils import (
    configure_dataset_type,
    get_datalake_and_tag,
    initialize_coco_data,
    upload_images,
)


def upload_dataset_context_based_on_type(
    dataset_context: CocoDatasetContext,
    datalake: Datalake,
    data_tag: str,
    use_id: bool,
    fail_on_asset_not_found: bool,
):
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


@step
def upload_dataset_context(
    dataset_context: CocoDatasetContext,
    datalake: Datalake,
    data_tag: str,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    datalake, data_tag = get_datalake_and_tag(
        context=context, datalake=datalake, data_tag=data_tag
    )
    dataset_context = initialize_coco_data(dataset_context=dataset_context)
    annotations = dataset_context.coco_data.get("annotations", [])
    if annotations:
        configure_dataset_type(dataset_context=dataset_context, annotations=annotations)

        upload_dataset_context_based_on_type(
            dataset_context=dataset_context,
            datalake=datalake,
            data_tag=data_tag,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
    else:
        upload_images(
            dataset_context=dataset_context, datalake=datalake, data_tag=data_tag
        )
