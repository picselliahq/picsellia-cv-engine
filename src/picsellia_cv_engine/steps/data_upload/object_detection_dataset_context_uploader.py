from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data_upload.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)


@step
def upload_object_detection_dataset_context(dataset_context: CocoDatasetContext):
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    uploader = ObjectDetectionDatasetContextUploader(
        client=context.client,
        dataset_context=dataset_context,
        datalake=context.processing_parameters.datalake,
        data_tags=[
            context.processing_parameters.data_tag,
            dataset_context.dataset_version.version,
        ],
    )
    uploader.upload_dataset_context()
