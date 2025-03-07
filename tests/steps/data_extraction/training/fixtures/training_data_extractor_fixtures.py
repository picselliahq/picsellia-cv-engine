from collections.abc import Callable

import pytest

from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.common.augmentation_parameters import (
    AugmentationParameters,
)
from picsellia_cv_engine.models.parameters.common.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.models.parameters.common.hyper_parameters import (
    HyperParameters,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def log_data() -> dict:
    return {
        "epochs": 3,
        "batch_size": 4,
        "image_size": 256,
    }


@pytest.fixture
def mock_picsellia_training_context(
    api_token: str,
    host: str,
    mock_experiment: Callable,
    log_data: dict,
) -> Callable:
    def _mock_picsellia_training_context(
        experiment_name: str,
        datasets_metadata: list[DatasetTestMetadata],
        hyperparameters_cls: type[HyperParameters],
        augmentation_parameters_cls: type[AugmentationParameters],
        export_parameters_cls: type[ExportParameters],
    ) -> PicselliaTrainingContext:
        experiment = mock_experiment(
            experiment_name=experiment_name, datasets_metadata=datasets_metadata
        )
        experiment.log_parameters(parameters=log_data)

        return PicselliaTrainingContext(
            hyperparameters_cls=hyperparameters_cls,
            augmentation_parameters_cls=augmentation_parameters_cls,
            export_parameters_cls=export_parameters_cls,
            api_token=api_token,
            host=host,
            experiment_id=experiment.id,
        )

    return _mock_picsellia_training_context
