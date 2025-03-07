import os

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.picsellia_datalake_processing_context import (
    PicselliaDatalakeProcessingContext,
)
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.dataset.datalake_collection import DatalakeCollection
from picsellia_cv_engine.models.dataset.datalake_context import DatalakeContext
from picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)


def get_destination_path(job_id: str | None) -> str:
    """
    Generates a destination path based on the current working directory and a job ID.

    Args:
        job_id (Optional[str]): The ID of the current job. If None, defaults to "current_job".

    Returns:
        str: The generated file path for the job.
    """
    if not job_id:
        return os.path.join(os.getcwd(), "current_job")
    return os.path.join(os.getcwd(), str(job_id))


@step
def get_processing_dataset_context(
    skip_asset_listing: bool = False,
) -> CocoDatasetContext:
    """
    Extracts a dataset context from a processing job, preparing it for further processing.

    This function retrieves the active processing context from the pipeline, uses the input dataset version,
    and creates a `DatasetContext`. The dataset context includes all necessary assets (e.g., images) and
    annotations (e.g., COCO format) required for processing. It downloads the assets and annotations into a
    destination folder based on the current job ID.

    Args:
        skip_asset_listing (bool): Whether to skip listing the dataset's assets during the download process. Defaults to False.

    Returns:
        CocoDatasetContext: The dataset context prepared for processing, including all downloaded assets and annotations.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    dataset_context = CocoDatasetContext(
        dataset_name="input",
        dataset_version=context.input_dataset_version,
        assets=context.input_dataset_version.list_assets(),
        labelmap=None,
    )
    destination_path = get_destination_path(context.job_id)

    dataset_context.download_assets(
        destination_dir=os.path.join(
            destination_path, "images", dataset_context.dataset_name
        ),
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )
    dataset_context.download_annotations(
        destination_dir=os.path.join(
            destination_path, "annotations", dataset_context.dataset_name
        ),
        use_id=True,
    )

    return dataset_context


@step
def get_processing_dataset_collection(
    skip_asset_listing: bool = False,
) -> DatasetCollection[CocoDatasetContext]:
    """
    Extracts a dataset collection from a processing job, preparing it for further processing.

    This function retrieves the active processing context from the pipeline, initializes a
    `ProcessingDatasetCollectionExtractor` with the input and output dataset versions, and downloads
    the necessary assets and annotations for both input and output datasets. It prepares the dataset collection
    and stores them in a specified destination folder.

    Args:
        skip_asset_listing (bool): Whether to skip listing the dataset's assets during the download process. Defaults to False.

    Returns:
        DatasetCollection: The dataset collection prepared for processing, including all downloaded assets and annotations.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    input_dataset_context = CocoDatasetContext(
        dataset_name="input",
        dataset_version=context.input_dataset_version,
        assets=context.input_dataset_version.list_assets(),
        labelmap=None,
    )
    output_dataset_context = CocoDatasetContext(
        dataset_name="output",
        dataset_version=context.output_dataset_version,
        assets=None,
        labelmap=None,
    )
    dataset_collection = DatasetCollection(
        [input_dataset_context, output_dataset_context]
    )
    destination_path = get_destination_path(context.job_id)
    dataset_collection.download_all(
        images_destination_dir=os.path.join(destination_path, "images"),
        annotations_destination_dir=os.path.join(destination_path, "annotations"),
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )
    return dataset_collection


@step
def processing_datalake_extractor() -> DatalakeContext | DatalakeCollection:
    context: PicselliaDatalakeProcessingContext = Pipeline.get_active_context()
    input_datalake_context = DatalakeContext(
        datalake_name="input",
        datalake=context.input_datalake,
        destination_path=get_destination_path(context.job_id),
        data_ids=context.data_ids,
        use_id=context.use_id,
    )
    if context.output_datalake:
        output_datalake_context = DatalakeContext(
            datalake_name="output",
            datalake=context.output_datalake,
            destination_path=get_destination_path(context.job_id),
            use_id=context.use_id,
        )
        datalake_collection = DatalakeCollection(
            input_datalake_context=input_datalake_context,
            output_datalake_context=output_datalake_context,
        )
        datalake_collection.download_all()
        return datalake_collection
    else:
        input_datalake_context.download_data(image_dir=input_datalake_context.image_dir)
        return input_datalake_context
