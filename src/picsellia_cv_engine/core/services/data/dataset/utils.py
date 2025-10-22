import logging
import os
from collections.abc import Sequence
from uuid import UUID

from picsellia.exceptions import NoDataError

from picsellia_cv_engine.core import DatasetCollection
from picsellia_cv_engine.core.contexts import (
    LocalDatasetProcessingContext,
    LocalTrainingContext,
    PicselliaDatasetProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.services.data.dataset.loader import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.core.services.data.dataset.validator.utils import (
    get_dataset_validator,
)
from picsellia_cv_engine.core.services.utils.dataset_logging import log_labelmap

logger = logging.getLogger(__name__)

# ----------------------------- Internals / helpers -----------------------------


def _dataset_images_dir(base: str) -> str:
    return os.path.join(base, "images")


def _dataset_ann_dir(base: str, ann_dir_name: str) -> str:
    return os.path.join(base, ann_dir_name)


def _prefetch_assets_for_context(
    context: PicselliaDatasetProcessingContext | LocalDatasetProcessingContext,
    dataset_version,
):
    """
    Return a MultiAsset for the given dataset_version:
      - If the context exposes `asset_ids`, call `list_assets(ids=asset_ids)`.
      - Otherwise, call `list_assets()`.

    If no assets are found, log a warning and re-raise NoDataError (so callers can decide).
    """
    asset_ids: Sequence[str | UUID] | None = getattr(context, "asset_ids", None)

    try:
        if asset_ids:
            # Narrow to the requested assets
            return dataset_version.list_assets(ids=list(asset_ids))
        # Full listing
        return dataset_version.list_assets()
    except NoDataError as e:
        if asset_ids:
            logger.warning(
                "No assets found for requested asset_ids (n=%d). Falling back raised.",
                len(asset_ids),
            )
        else:
            logger.warning("No assets found in dataset version.")
        raise e


def _load_training_datasets(
    *,
    context: PicselliaTrainingContext | LocalTrainingContext,
    dataset_cls: type[TBaseDataset],
    ann_dir_name: str,
    use_id: bool,
) -> DatasetCollection[TBaseDataset]:
    """
    Loading logic for training contexts (Picsellia/Local).
    """
    extractor = TrainingDatasetCollectionExtractor(
        experiment=context.experiment,
        train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
    )

    dataset_collection: DatasetCollection[TBaseDataset] = (
        extractor.get_dataset_collection(
            context_class=dataset_cls,
            random_seed=context.hyperparameters.seed,
        )
    )

    # Log labelmap
    log_labelmap(
        labelmap=dataset_collection["train"].labelmap,
        experiment=context.experiment,
        log_name="labelmap",
    )

    # Directory layout & downloads
    dataset_collection.dataset_path = os.path.join(context.working_dir, "dataset")
    dataset_collection.download_all(
        images_destination_dir=_dataset_images_dir(dataset_collection.dataset_path),
        annotations_destination_dir=_dataset_ann_dir(
            dataset_collection.dataset_path, ann_dir_name
        ),
        use_id=use_id,
        skip_asset_listing=False,  # always list for training
    )
    return dataset_collection


def _load_processing_datasets(
    *,
    context: PicselliaDatasetProcessingContext | LocalDatasetProcessingContext,
    dataset_cls: type[TBaseDataset],
    ann_dir_name: str,
    use_id: bool,
    skip_asset_listing: bool,
) -> DatasetCollection[TBaseDataset] | TBaseDataset:
    """
    Loading logic for processing contexts (Picsellia/Local).

    Handles:
      - input != output  -> DatasetCollection(input, output)
      - input == output or output missing -> single dataset (input)
    """
    in_id = context.input_dataset_version_id
    out_id = context.output_dataset_version_id

    # input & output are distinct -> collection
    if in_id and out_id and in_id != out_id:
        # Prefetch filtered assets (MultiAsset) for input if asset_ids were provided
        try:
            input_assets = _prefetch_assets_for_context(
                context, context.input_dataset_version
            )
        except NoDataError:
            # Keep going: create dataset with no preloaded assets; downloads may still succeed
            logger.warning("Proceeding without preloaded assets for input dataset.")
            input_assets = None

        input_dataset = dataset_cls(
            name="input",
            dataset_version=context.input_dataset_version,
            assets=input_assets,  # MultiAsset or None
            labelmap=None,
        )
        output_dataset = dataset_cls(
            name="output",
            dataset_version=context.output_dataset_version,
            assets=None,  # we don't need to prefetch assets for the output DV
            labelmap=None,
        )

        dataset_collection = DatasetCollection([input_dataset, output_dataset])
        dataset_collection.download_all(
            images_destination_dir=_dataset_images_dir(context.working_dir),
            annotations_destination_dir=_dataset_ann_dir(
                context.working_dir, ann_dir_name
            ),
            use_id=use_id,
            skip_asset_listing=skip_asset_listing,
        )
        return dataset_collection

    # otherwise: single input dataset
    if in_id and (in_id == out_id or not out_id):
        try:
            assets = _prefetch_assets_for_context(
                context, context.input_dataset_version
            )
        except NoDataError:
            logger.warning(
                "Proceeding without preloaded assets for single input dataset."
            )
            assets = None

        dataset: TBaseDataset = dataset_cls(
            name="input",
            dataset_version=context.input_dataset_version,
            assets=assets,  # MultiAsset or None
            labelmap=None,
        )
        dataset.download_assets(
            destination_dir=os.path.join(context.working_dir, "images", dataset.name),
            use_id=use_id,
            skip_asset_listing=skip_asset_listing,
        )
        dataset.download_annotations(
            destination_dir=os.path.join(
                context.working_dir, ann_dir_name, dataset.name
            ),
            use_id=use_id,
        )
        return dataset

    raise ValueError("No datasets found in the processing context.")


def load_datasets_impl_generic(
    *,
    context: PicselliaTrainingContext
    | LocalTrainingContext
    | PicselliaDatasetProcessingContext
    | LocalDatasetProcessingContext,
    dataset_cls: type[TBaseDataset],
    ann_dir_name: str,
    use_id: bool,
    skip_asset_listing: bool,
) -> DatasetCollection[TBaseDataset] | TBaseDataset:
    """
    Generic implementation shared by COCO/YOLO.
    Clear orchestration by context type, minimal duplication.
    """
    # Training contexts
    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        return _load_training_datasets(
            context=context,
            dataset_cls=dataset_cls,
            ann_dir_name=ann_dir_name,
            use_id=use_id,
        )

    # Processing contexts
    if isinstance(
        context, PicselliaDatasetProcessingContext | LocalDatasetProcessingContext
    ):
        return _load_processing_datasets(
            context=context,
            dataset_cls=dataset_cls,
            ann_dir_name=ann_dir_name,
            use_id=use_id,
            skip_asset_listing=skip_asset_listing,
        )

    raise ValueError(f"Unsupported context type: {type(context)}")


# --------------------------------- Validation ---------------------------------


def validate_dataset_impl(
    dataset: TBaseDataset | DatasetCollection, fix_annotation: bool = False
):
    """
    Validate a single dataset or each split of a DatasetCollection.
    Logs failures per split for easier troubleshooting.
    """
    if not isinstance(dataset, DatasetCollection):
        validator = get_dataset_validator(
            dataset=dataset, fix_annotation=fix_annotation
        )
        if validator:
            validator.validate()
        return

    # DatasetCollection
    for name, ds in dataset.datasets.items():
        try:
            validator = get_dataset_validator(dataset=ds, fix_annotation=fix_annotation)
            if validator:
                validator.validate()
            else:
                logger.info(f"Skipping validation for dataset '{name}'.")
        except Exception as e:
            logger.error(f"Validation failed for dataset '{name}': {str(e)}")
