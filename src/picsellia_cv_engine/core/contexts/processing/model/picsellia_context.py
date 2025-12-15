from typing import Any, Generic, TypeVar

from picsellia import ModelVersion

from picsellia_cv_engine.core.contexts.processing.common.picsellia_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaModelProcessingContext(PicselliaProcessingContext, Generic[TParameters]):
    def _load_inputs(self, **kwargs: Any) -> None:
        self._model_version_id = self.inputs.get("input_model_version_id")

        if self._model_version_id:
            self.model_version = self.get_model_version()

    @property
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

    def get_model_version(self) -> ModelVersion:
        return self.client.get_model_version_by_id(self.model_version_id)
