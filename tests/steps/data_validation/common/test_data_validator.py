from collections.abc import Callable
from unittest.mock import patch

import pytest
from picsellia.types.enums import InferenceType
from steps.data_validation.common.data_validator import training_data_validator


class TestDataValidator:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_training_data_validator_validator(
        self, mock_dataset_collection: Callable, dataset_type: InferenceType
    ):
        with patch(
            "models.steps.data_validation.common.dataset_collection_validator.DatasetCollectionValidator.validate"
        ) as mocked_validate:
            dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
            training_data_validator.entrypoint(dataset_collection=dataset_collection)
            assert mocked_validate.call_count == 1
