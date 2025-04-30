import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, YoloDataset
from picsellia_cv_engine.core.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.services.data.dataset.loader import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.core.services.utils.dataset_logging import log_labelmap


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
    return load_coco_datasets_impl(
        context=context, skip_asset_listing=skip_asset_listing
    )


@step
def load_yolo_datasets(
    skip_asset_listing: bool = False,
) -> DatasetCollection[YoloDataset] | YoloDataset:
    context = Pipeline.get_active_context()
    return load_yolo_datasets_impl(
        context=context, skip_asset_listing=skip_asset_listing
    )


def load_coco_datasets_impl(
    context, skip_asset_listing: bool
) -> DatasetCollection[CocoDataset] | CocoDataset:
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

        dataset_collection.dataset_path = os.path.join(context.working_dir, "dataset")

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
        if (
            context.input_dataset_version_id
            and context.output_dataset_version_id
            and not context.input_dataset_version_id
            == context.output_dataset_version_id
        ):
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
            dataset_collection.download_all(
                images_destination_dir=os.path.join(context.working_dir, "images"),
                annotations_destination_dir=os.path.join(
                    context.working_dir, "annotations"
                ),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            return dataset_collection

        # If only input dataset is available
        elif (
            context.input_dataset_version_id
            and context.input_dataset_version_id == context.output_dataset_version_id
        ) or (
            context.input_dataset_version_id and not context.output_dataset_version_id
        ):
            dataset = CocoDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )

            dataset.download_assets(
                destination_dir=os.path.join(
                    context.working_dir, "images", dataset.name
                ),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            dataset.download_annotations(
                destination_dir=os.path.join(
                    context.working_dir, "annotations", dataset.name
                ),
                use_id=True,
            )

            return dataset

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")


def load_yolo_datasets_impl(
    context, skip_asset_listing: bool
) -> DatasetCollection[YoloDataset] | YoloDataset:
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

        dataset_collection.dataset_path = os.path.join(context.working_dir, "dataset")

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
        if (
            context.input_dataset_version_id
            and context.output_dataset_version_id
            and not context.input_dataset_version_id
            == context.output_dataset_version_id
        ):
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
            dataset_collection.download_all(
                images_destination_dir=os.path.join(context.working_dir, "images"),
                annotations_destination_dir=os.path.join(context.working_dir, "labels"),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            return dataset_collection

        # If only input dataset is available
        elif (
            context.input_dataset_version_id
            and context.input_dataset_version_id == context.output_dataset_version_id
        ) or (
            context.input_dataset_version_id and not context.output_dataset_version_id
        ):
            dataset = YoloDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )

            dataset.download_assets(
                destination_dir=os.path.join(
                    context.working_dir, "images", dataset.name
                ),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            dataset.download_annotations(
                destination_dir=os.path.join(
                    context.working_dir, "labels", dataset.name
                ),
                use_id=True,
            )

            return dataset

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")
