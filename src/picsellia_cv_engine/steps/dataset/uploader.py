from picsellia import Datalake

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data.dataset.uploader.utils import (
    configure_dataset_type,
    get_datalake_and_tag,
    initialize_coco_data,
    upload_annotations_based_on_inference_type,
    upload_dataset_context_based_on_type,
    upload_images,
)


@step
def upload_full_dataset(
    dataset_context: CocoDatasetContext,
    datalake: Datalake | None = None,
    data_tag: str | None = None,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    """
    Upload both images and annotations for a dataset.

    This step handles the entire dataset upload process, including:

    - Configuring the dataset type based on annotations
    - Uploading images and annotations depending on the dataset's inference type

    Args:
        dataset_context (CocoDatasetContext): The dataset containing images and annotations to upload.
        datalake (Optional[Datalake]): The Datalake instance to upload to. If not provided, it is retrieved from the processing context.
        data_tag (Optional[str]): The tag associated with the dataset upload. If not provided, it is retrieved from the context.
        use_id (bool): Flag to indicate whether to use asset IDs during upload. Defaults to True.
        fail_on_asset_not_found (bool): Flag to determine if the upload should fail if an asset is not found. Defaults to True.
    """
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


@step
def upload_dataset_images(
    dataset_context: CocoDatasetContext,
    datalake: Datalake | None = None,
    data_tag: str | None = None,
) -> None:
    """
    Upload only the images from a dataset.

    This step focuses on uploading image assets associated with the provided dataset context.

    Args:
        dataset_context (CocoDatasetContext): The dataset containing images to upload.
        datalake (Optional[Datalake]): The Datalake instance to upload the images to. If not provided, it is retrieved from the processing context.
        data_tag (Optional[str]): The tag associated with the dataset upload. If not provided, it is retrieved from the context.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    datalake, data_tag = get_datalake_and_tag(
        context=context, datalake=datalake, data_tag=data_tag
    )

    upload_images(dataset_context=dataset_context, datalake=datalake, data_tag=data_tag)


@step
def upload_dataset_annotations(
    dataset_context: CocoDatasetContext,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    """
    Upload only the annotations for a dataset.

    This step handles the upload of annotations, which are configured based on the dataset context. If annotations exist,
    the dataset type is configured, and the annotations are uploaded based on the inference type.

    Args:
        dataset_context (CocoDatasetContext): The dataset containing annotations to upload.
        use_id (bool): Flag to indicate whether to use asset IDs during upload. Defaults to True.
        fail_on_asset_not_found (bool): Flag to determine if the upload should fail if an asset is not found. Defaults to True.
    """

    dataset_context = initialize_coco_data(dataset_context=dataset_context)
    annotations = dataset_context.coco_data.get("annotations", [])

    if annotations:
        configure_dataset_type(dataset_context=dataset_context, annotations=annotations)
        upload_annotations_based_on_inference_type(
            dataset_context=dataset_context,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
