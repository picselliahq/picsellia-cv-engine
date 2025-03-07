import os

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data_extraction.training_dataset_collection_extractor import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.models.utils.dataset_logging import (
    log_labelmap,
)


@step
def get_coco_dataset_collection() -> DatasetCollection[CocoDatasetContext]:
    """
    Extracts datasets from an experiment and prepares them for training, validation, and testing.

    This function retrieves the active training context from the pipeline and uses it to initialize a
    `TrainingDatasetCollectionExtractor` with the current experiment and the training split ratio from the
    hyperparameters. It retrieves a `DatasetCollection` of datasets ready for use in training, validation,
    and testing, downloading all necessary assets and annotations.

    The function also logs the labelmap and the objects distribution for each dataset split in the collection,
    facilitating data analysis and tracking in the experiment.

    Returns:
        DatasetCollection: A collection of dataset contexts prepared for training, validation, and testing,
        with all necessary assets and annotations downloaded.

    Raises:
        ResourceNotFoundError: If any of the expected dataset splits (train, validation, test) are not found in the experiment.
        RuntimeError: If an invalid number of datasets are attached to the experiment.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

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
