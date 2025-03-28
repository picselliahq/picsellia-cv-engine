import os
from typing import Any, Generic, TypeVar

import picsellia  # type: ignore
from picsellia import ModelVersion

from picsellia_cv_engine.models.contexts import PicselliaContext
from picsellia_cv_engine.models.parameters import ExportParameters, Parameters

TParameters = TypeVar("TParameters", bound=Parameters)
TExportParameters = TypeVar("TExportParameters", bound=ExportParameters)


class PicselliaModelProcessingContext(
    PicselliaContext, Generic[TParameters, TExportParameters]
):
    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        export_parameters_cls: type[TExportParameters],
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        job_id: str | None = None,
        use_id: bool | None = True,
        download_annotations: bool | None = True,
    ):
        super().__init__(api_token, host, organization_id)

        self.job_id = job_id or os.environ.get("job_id")
        if not self.job_id:
            raise ValueError(
                "Job ID not provided. Please provide it as an argument or set the 'job_id' environment variable."
            )

        self.job = self._initialize_job()
        self.job_type = self.job.sync()["type"]

        self.job_context = self._initialize_job_context()

        self._model_version_id = self.job_context.get("input_model_version_id")

        if self._model_version_id:
            self.model_version = self.get_model_version()

        self.use_id = use_id
        self.download_annotations = download_annotations

        parameters_log_data = self.job_context["parameters"]

        self.processing_parameters = processing_parameters_cls(
            log_data=parameters_log_data
        )
        self.export_parameters = export_parameters_cls(log_data=parameters_log_data)

    @property
    def model_version_id(self) -> str | None:
        if not self._model_version_id:
            raise ValueError(
                "Model version ID not found. Please ensure the job is correctly configured."
            )

        return self._model_version_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_id": self.job_id,
            },
            "model_version_id": self.model_version_id,
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
            "export_parameters": self._process_parameters(
                parameters_dict=self.export_parameters.to_dict(),
                defaulted_keys=self.export_parameters.defaulted_keys,
            ),
        }

    def _initialize_job_context(self) -> dict[str, Any]:
        """Initializes the context by fetching the necessary information from the job."""
        job_context = self.job.sync()["model_version_processing_job"]

        return job_context

    def _initialize_job(self) -> picsellia.Job:
        """
        Fetches the job from Picsellia using the job ID.

        The Job, in a Picsellia processing context,
        is the entity that contains all the information needed to run a processing job.

        Returns:
            The job fetched from Picsellia.
        """
        return self.client.get_job_by_id(self.job_id)

    def get_model_version(self) -> ModelVersion:
        """
        Fetches the model version from Picsellia using the model version ID.

        The ModelVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a model.

        Returns:
            The model version fetched from Picsellia.
        """
        return self.client.get_model_version_by_id(self.model_version_id)
