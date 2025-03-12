import logging

from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.utils import (
    get_validator_for_dataset_context,
)

logger = logging.getLogger(__name__)


@step
def validate_dataset(
    dataset: TBaseDatasetContext | DatasetCollection, fix_annotation: bool = False
):
    """
    Validates a dataset or a dataset collection to ensure data integrity and correctness.

    This function checks each dataset in a collection or a single dataset for any issues. If annotation errors are found,
    it can attempt to fix them based on the provided `fix_annotation` flag. If validation fails for any dataset,
    an error is logged. The validation process is skipped for datasets without a validator.

    Args:
        dataset (Union[TBaseDatasetContext, DatasetCollection]):
            The dataset or dataset collection to validate. If a `DatasetCollection` is provided, each individual dataset
            within the collection will be validated.
        fix_annotation (bool, optional):
            Flag to indicate whether to attempt fixing annotation errors. Defaults to `False`. If set to `True`,
            the function will try to correct any found annotation issues during validation.

    Raises:
        Exception: If validation fails for a dataset in the collection, an error is logged, but the process continues.
    """
    validators = {}

    if not isinstance(dataset, DatasetCollection):
        validator = get_validator_for_dataset_context(dataset)
        if validator:
            validator.validate(fix_annotation=fix_annotation)

    for dataset_name, dataset_context in dataset.items():
        try:
            validator = get_validator_for_dataset_context(dataset_context)
            if validator:
                validator.validate(fix_annotation=fix_annotation)
                validators[dataset_name] = validator
            else:
                logger.info(f"Skipping validation for dataset '{dataset_name}'.")
        except Exception as e:
            logger.error(f"Validation failed for dataset '{dataset_name}': {str(e)}")
