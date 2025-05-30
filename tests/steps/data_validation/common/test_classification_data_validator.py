from collections.abc import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType
from steps.data_validation.common.classification_data_validator import (
    classification_dataset_collection_validator,
)


class TestDataValidator:
    def test_data_validator(self, mock_dataset_collection: Callable):
        with patch(
            "models.steps.data_validation.common.classification_dataset_context_validator"
            ".ClassificationDatasetContextValidator"
            ".validate"
        ) as mocked_validate:
            dataset_collection = mock_dataset_collection(
                dataset_type=InferenceType.CLASSIFICATION
            )
            classification_dataset_collection_validator.entrypoint(
                dataset_collection=dataset_collection
            )
            assert mocked_validate.call_count == 3
