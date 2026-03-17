from typing import Any, Generic, TypeVar

from deprecation import deprecated
from picsellia import ModelVersion
from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.contexts.processing.common.local_picsellia_context import (
    PicselliaLocalProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class LocalModelProcessingContext(
    PicselliaLocalProcessingContext, Generic[TParameters]
):
    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        processing_parameters: dict[str, Any] | None = None,
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        job_type: ProcessingType | None = None,
        target_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ):
        self.job_type = job_type
        super().__init__(
            processing_parameters_cls=processing_parameters_cls,
            parameters_dict=processing_parameters,
            api_token=api_token,
            host=host,
            organization_id=organization_id,
            organization_name=organization_name,
            target_id=target_id,
            inputs=inputs,
            working_dir=working_dir,
        )
        self.target = self.client.get_model_version_by_id(id=self.target_id)

    def _load_legacy_inputs(self, **kwargs) -> None:
        self._model_version_id = self.inputs.get("input_model_version_id")
        self.model_version = self.get_model_version()

    @property
    @deprecated(
        details="model_version_id will be removed in a future version. Use the new input system instead."
    )
    def model_version_id(self) -> str:
        if not self._model_version_id:
            raise ValueError(
                "Model version ID not found. Please ensure the job is correctly configured."
            )
        return self._model_version_id

    @deprecated(
        details="get_model_version will be removed in a future version. Use the new input system instead."
    )
    def get_model_version(self) -> ModelVersion:
        return self.client.get_model_version_by_id(self._model_version_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary for logging or serialization."""
        base = super().to_dict()
        base.update(
            {
                "job_type": self.job_type,
                "model_version_id": self.model_version_id,
            }
        )
        return base
