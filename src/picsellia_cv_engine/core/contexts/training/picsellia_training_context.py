import os
from typing import Any, Generic, TypeVar

from picsellia import Experiment  # type: ignore

from picsellia_cv_engine.core.contexts import PicselliaContext
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
    HyperParameters,
)

THyperParameters = TypeVar("THyperParameters", bound=HyperParameters)
TAugmentationParameters = TypeVar(
    "TAugmentationParameters", bound=AugmentationParameters
)
TExportParameters = TypeVar("TExportParameters", bound=ExportParameters)


class PicselliaTrainingContext(
    PicselliaContext,
    Generic[THyperParameters, TAugmentationParameters, TExportParameters],
):
    def __init__(
        self,
        hyperparameters_cls: type[THyperParameters],
        augmentation_parameters_cls: type[TAugmentationParameters],
        export_parameters_cls: type[TExportParameters],
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        experiment_id: str | None = None,
    ):
        super().__init__(api_token, host, organization_id)
        self.experiment_id = experiment_id or os.getenv("experiment_id")

        if not self.experiment_id:
            raise ValueError(
                "Experiment ID not provided. Please provide it as an argument "
                "or set the 'experiment_id' environment variable."
            )

        self.experiment = self._initialize_experiment()
        parameters_log_data = self.experiment.get_log("parameters").data

        self.hyperparameters = hyperparameters_cls(log_data=parameters_log_data)
        self.augmentation_parameters = augmentation_parameters_cls(
            log_data=parameters_log_data
        )
        self.export_parameters = export_parameters_cls(log_data=parameters_log_data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "organization_name": self.organization_name,
                "experiment_id": self.experiment_id,
            },
            "hyperparameters": self._process_parameters(
                parameters_dict=self.hyperparameters.to_dict(),
                defaulted_keys=self.hyperparameters.defaulted_keys,
            ),
            "augmentation_parameters": self._process_parameters(
                parameters_dict=self.augmentation_parameters.to_dict(),
                defaulted_keys=self.augmentation_parameters.defaulted_keys,
            ),
            "export_parameters": self._process_parameters(
                parameters_dict=self.export_parameters.to_dict(),
                defaulted_keys=self.export_parameters.defaulted_keys,
            ),
        }

    def _initialize_experiment(self) -> Experiment:
        """Fetches the experiment from Picsellia using the experiment ID.

        The experiment, in a Picsellia training context,
        is the entity that contains all the information needed to train a models.

        Returns:
            The experiment fetched from Picsellia.
        """
        return self.client.get_experiment_by_id(self.experiment_id)
