from typing import Any, Generic, TypeVar
from uuid import UUID

import requests
from picsellia import Datalake, ModelVersion

from picsellia_cv_engine.core.contexts.processing.common.picsellia_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaDatalakeProcessingContext(
    PicselliaProcessingContext, Generic[TParameters]
):
    """
    Context for running Picsellia datalake processing jobs.
    """

    def _load_inputs(self, **kwargs: Any) -> None:
        processing_name = self.job_context["processing_name"]
        self.processing_type = self.client.get_processing(name=processing_name).type

        self._model_version_id = self.inputs.get("model_version_id")
        self._input_datalake_id = self.inputs.get("input_datalake_id")
        self._output_datalake_id = self.inputs.get("output_datalake_id")

        if not self._input_datalake_id:
            raise ValueError("Input datalake ID not found.")

        self.input_datalake = self.get_datalake(self._input_datalake_id)
        self.output_datalake = (
            self.get_datalake(self._output_datalake_id)
            if self._output_datalake_id
            else None
        )

        self.model_version = (
            self.get_model_version(model_version_id=self._model_version_id)
            if self._model_version_id
            else None
        )
        self.data_ids = self.get_data_ids()

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "model_version_id": self.model_version_id,
                "input_datalake_id": self._input_datalake_id,
                "output_datalake_id": self._output_datalake_id,
            }
        )
        return base

    def get_datalake(self, datalake_id: str) -> Datalake:
        return self.client.get_datalake(id=datalake_id)

    def get_model_version(self, model_version_id: str) -> ModelVersion:
        return self.client.get_model_version_by_id(self.model_version_id)

    def get_data_ids(self) -> list[UUID]:
        """
        Retrieve data IDs from the job payload.

        Raises:
            ValueError: If the payload URL is missing or invalid.
        """
        if not self.payload_presigned_url:
            raise ValueError("Payload presigned URL not found.")
        payload = requests.get(self.payload_presigned_url).json()
        return [UUID(data_id) for data_id in payload["data_ids"]]
