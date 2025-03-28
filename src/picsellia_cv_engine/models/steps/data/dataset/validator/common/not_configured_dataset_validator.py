from picsellia_cv_engine.models.data import TBaseDataset
from picsellia_cv_engine.models.steps.data.dataset.validator.common.dataset_validator import (
    DatasetValidator,
)


class NotConfiguredDatasetValidator(DatasetValidator[TBaseDataset]):
    def validate(self):
        """
        Validate the dataset.

        Raises:
            ValueError: If the dataset is not valid.
        """
        super().validate()
