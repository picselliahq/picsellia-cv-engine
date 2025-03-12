import os

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.local_picsellia_processing_context import (
    LocalPicselliaProcessingContext,
)
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.contexts.training.local_picsellia_training_context import (
    LocalPicselliaTrainingContext,
)
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data.dataset.loader.training_dataset_collection_extractor import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.models.steps.data.utils import get_destination_path
from picsellia_cv_engine.models.utils.dataset_logging import (
    log_labelmap,
)


@step
def load_coco_datasets(
    skip_asset_listing: bool = False,
) -> DatasetCollection[CocoDatasetContext] | CocoDatasetContext:
    """
    A step for loading COCO datasets based on the current pipeline context (training or processing).

    This function adapts to different contexts and loads datasets accordingly:
    - **Training Contexts**: Loads datasets for training, validation, and testing splits.
    - **Processing Contexts**: Loads either a single dataset or multiple datasets depending on the context.

    Args:
        skip_asset_listing (bool, optional): Flag to determine whether to skip listing dataset assets before downloading.
            Default is `False`. This is applicable only for processing contexts.

    Returns:
        Union[DatasetCollection[CocoDatasetContext], CocoDatasetContext]: The loaded dataset(s) based on the context.

            - For **Training Contexts**: Returns a `DatasetCollection[CocoDatasetContext]` containing training, validation,
              and test datasets.
            - For **Processing Contexts**:
                - If both input and output datasets are available, returns a `DatasetCollection[CocoDatasetContext]`.
                - If only an input dataset is available, returns a single `CocoDatasetContext` for the input dataset.

    Raises:
        ValueError:
            - If no datasets are found in the processing context.
            - If the context type is unsupported (neither training nor processing).

    Example:
        - In a **Training Context**, the function loads and prepares datasets for training, validation, and testing.
        - In a **Processing Context**, it loads the input and output datasets (if available) or just the input dataset.
    """
    context = Pipeline.get_active_context()

    # Training Context Handling
    if isinstance(context, PicselliaTrainingContext | LocalPicselliaTrainingContext):
        dataset_collection_extractor = TrainingDatasetCollectionExtractor(
            experiment=context.experiment,
            train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
        )

        coco_dataset_collection = dataset_collection_extractor.get_dataset_collection(
            context_class=CocoDatasetContext,
            random_seed=context.hyperparameters.seed,
        )

        log_labelmap(
            labelmap=coco_dataset_collection["train"].labelmap,
            experiment=context.experiment,
            log_name="labelmap",
        )

        coco_dataset_collection.dataset_path = os.path.join(
            os.getcwd(), context.experiment.name, "dataset"
        )

        coco_dataset_collection.download_all(
            images_destination_dir=os.path.join(
                coco_dataset_collection.dataset_path, "images"
            ),
            annotations_destination_dir=os.path.join(
                coco_dataset_collection.dataset_path, "annotations"
            ),
            use_id=True,
            skip_asset_listing=False,
        )

        return coco_dataset_collection

    # Processing Context Handling
    elif isinstance(
        context, PicselliaProcessingContext | LocalPicselliaProcessingContext
    ):
        # If both input and output datasets are available
        if context.input_dataset_version_id and context.output_dataset_version_id:
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
                annotations_destination_dir=os.path.join(
                    destination_path, "annotations"
                ),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            return dataset_collection

        # If only input dataset is available
        elif context.input_dataset_version_id:
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

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")
