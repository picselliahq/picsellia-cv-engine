from typing import Any, Generic, TypeVar

from deprecation import deprecated
from picsellia import ModelVersion

from picsellia_cv_engine.core.contexts.processing.common.picsellia_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaModelProcessingContext(PicselliaProcessingContext, Generic[TParameters]):
    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        job_id: str | None = None,
        use_id: bool | None = True,
        working_dir: str | None = None,
    ):
        super().__init__(
            processing_parameters_cls=processing_parameters_cls,
            api_token=api_token,
            host=host,
            organization_id=organization_id,
            organization_name=organization_name,
            job_id=job_id,
            use_id=use_id,
            working_dir=working_dir,
        )

        self.target = self.client.get_model_version_by_id(id=self.target_id)

    def _load_legacy_inputs(self) -> None:
        self._model_version_id = self.inputs.get("input_model_version_id")

        if self._model_version_id:
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

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"model_version_id": self.model_version_id})
        return base

    @deprecated(
        details="get_model_version will be removed in a future version. Use the new input system instead."
    )
    def get_model_version(self) -> ModelVersion:
        return self.client.get_model_version_by_id(self.model_version_id)
