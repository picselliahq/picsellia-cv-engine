import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.models import CocoDataset, DatasetCollection, YoloDataset
from picsellia_cv_engine.models.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.steps.data.dataset.loader import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.models.steps.data.utils import get_destination_path
from picsellia_cv_engine.models.utils.dataset_logging import (
    log_labelmap,
)


@step
def load_coco_datasets(
    skip_asset_listing: bool = False,
) -> DatasetCollection[CocoDataset] | CocoDataset:
    """
    A step for loading COCO datasets based on the current pipeline context (training or processing).

    This function adapts to different contexts and loads datasets accordingly:
    - **Training Contexts**: Loads datasets for training, validation, and testing splits.
    - **Processing Contexts**: Loads either a single dataset or multiple datasets depending on the context.

    Args:
        skip_asset_listing (bool, optional): Flag to determine whether to skip listing dataset assets before downloading.
            Default is `False`. This is applicable only for processing contexts.

    Returns:
        Union[DatasetCollection[CocoDataset], CocoDataset]: The loaded dataset(s) based on the context.

            - For **Training Contexts**: Returns a `DatasetCollection[CocoDataset]` containing training, validation,
              and test datasets.
            - For **Processing Contexts**:
                - If both input and output datasets are available, returns a `DatasetCollection[CocoDataset]`.
                - If only an input dataset is available, returns a single `CocoDataset` for the input dataset.

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
    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        dataset_collection_extractor = TrainingDatasetCollectionExtractor(
            experiment=context.experiment,
            train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
        )

        dataset_collection = dataset_collection_extractor.get_dataset_collection(
            context_class=CocoDataset,
            random_seed=context.hyperparameters.seed,
        )

        log_labelmap(
            labelmap=dataset_collection["train"].labelmap,
            experiment=context.experiment,
            log_name="labelmap",
        )

        dataset_collection.dataset_path = os.path.join(
            os.getcwd(), context.experiment.name, "dataset"
        )

        dataset_collection.download_all(
            images_destination_dir=os.path.join(
                dataset_collection.dataset_path, "images"
            ),
            annotations_destination_dir=os.path.join(
                dataset_collection.dataset_path, "annotations"
            ),
            use_id=True,
            skip_asset_listing=False,
        )

        return dataset_collection

    # Processing Context Handling
    elif isinstance(context, PicselliaProcessingContext | LocalProcessingContext):
        # If both input and output datasets are available
        if context.input_dataset_version_id and context.output_dataset_version_id:
            input_dataset = CocoDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )
            output_dataset = CocoDataset(
                name="output",
                dataset_version=context.output_dataset_version,
                assets=None,
                labelmap=None,
            )
            dataset_collection = DatasetCollection([input_dataset, output_dataset])
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
            dataset = CocoDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )
            destination_path = get_destination_path(context.job_id)

            dataset.download_assets(
                destination_dir=os.path.join(destination_path, "images", dataset.name),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            dataset.download_annotations(
                destination_dir=os.path.join(
                    destination_path, "annotations", dataset.name
                ),
                use_id=True,
            )

            return dataset

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")


@step
def load_yolo_datasets(
    skip_asset_listing: bool = False,
) -> DatasetCollection[YoloDataset] | YoloDataset:
    context = Pipeline.get_active_context()

    # Training Context Handling
    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        dataset_collection_extractor = TrainingDatasetCollectionExtractor(
            experiment=context.experiment,
            train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
        )

        dataset_collection = dataset_collection_extractor.get_dataset_collection(
            context_class=YoloDataset,
            random_seed=context.hyperparameters.seed,
        )

        log_labelmap(
            labelmap=dataset_collection["train"].labelmap,
            experiment=context.experiment,
            log_name="labelmap",
        )

        dataset_collection.dataset_path = os.path.join(
            os.getcwd(), context.experiment.name, "dataset"
        )

        dataset_collection.download_all(
            images_destination_dir=os.path.join(
                dataset_collection.dataset_path, "images"
            ),
            annotations_destination_dir=os.path.join(
                dataset_collection.dataset_path, "labels"
            ),
            use_id=True,
            skip_asset_listing=False,
        )

        return dataset_collection

    # Processing Context Handling
    elif isinstance(context, PicselliaProcessingContext | LocalProcessingContext):
        # If both input and output datasets are available
        if context.input_dataset_version_id and context.output_dataset_version_id:
            input_dataset = YoloDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )
            output_dataset = YoloDataset(
                name="output",
                dataset_version=context.output_dataset_version,
                assets=None,
                labelmap=None,
            )
            dataset_collection = DatasetCollection([input_dataset, output_dataset])
            destination_path = get_destination_path(context.job_id)
            dataset_collection.download_all(
                images_destination_dir=os.path.join(destination_path, "images"),
                annotations_destination_dir=os.path.join(destination_path, "labels"),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            return dataset_collection

        # If only input dataset is available
        elif context.input_dataset_version_id:
            dataset = YoloDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )
            destination_path = get_destination_path(context.job_id)

            dataset.download_assets(
                destination_dir=os.path.join(destination_path, "images", dataset.name),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            dataset.download_annotations(
                destination_dir=os.path.join(destination_path, "labels", dataset.name),
                use_id=True,
            )

            return dataset

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")
