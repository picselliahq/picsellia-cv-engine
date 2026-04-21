from typing import Any, Generic, TypeVar
from uuid import UUID

import requests
from deprecation import deprecated
from picsellia import DatasetVersion, ModelVersion
from picsellia.exceptions import ResourceConflictError

from picsellia_cv_engine.core.contexts.processing.common.picsellia_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaDatasetProcessingContext(
    PicselliaProcessingContext, Generic[TParameters]
):
    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        job_id: str | None = None,
        use_id: bool | None = True,
        download_annotations: bool | None = True,
        working_dir: str | None = None,
    ):
        self.download_annotations = download_annotations
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
        self.asset_ids = self.get_asset_ids()
        self.target = self.client.get_dataset_version_by_id(id=self.target_id)

    def get_asset_ids(self) -> list[UUID] | None:
        if self.payload_presigned_url:
            payload = requests.get(self.payload_presigned_url).json()
            return [UUID(asset_id) for asset_id in payload["asset_ids"]]
        return None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "model_version_id": self.model_version_id,
                "input_dataset_version_id": self.input_dataset_version_id,
                "output_dataset_version_id": str(self.output_dataset_version.id),
            }
        )
        return base

    def _load_legacy_inputs(self) -> None:
        self._model_version_id = self.inputs.get("model_version_id")
        self._input_dataset_version_id = self.target_id
        self._target_version_name = self.inputs.get("target_version_name")

        self.input_dataset_version = self.get_dataset_version(
            self.input_dataset_version_id
        )

        if self._target_version_name:
            self.output_dataset_version = self.get_or_create_target_dataset_version(
                input_dataset_version=self.input_dataset_version,
                target_version_name=self._target_version_name,
            )
        else:
            self.output_dataset_version = self.input_dataset_version

        if self._model_version_id:
            self.model_version = self.get_model_version()

    @property
    @deprecated(
        details="input_dataset_version_id will be removed in a future version. Use the new input system instead."
    )
    def input_dataset_version_id(self) -> str:
        return self._input_dataset_version_id

    @property
    @deprecated(
        details="model_version_id will be removed in a future version. Use the new input system instead."
    )
    def model_version_id(self) -> str | None:
        return self._model_version_id

    @deprecated(
        details="get_dataset_version will be removed in a future version. Use the new input system instead."
    )
    def get_dataset_version(self, dataset_version_id: str) -> DatasetVersion:
        return self.client.get_dataset_version_by_id(dataset_version_id)

    @deprecated(
        details="get_model_version will be removed in a future version. Use the new input system instead."
    )
    def get_model_version(self) -> ModelVersion:
        return self.client.get_model_version_by_id(self.model_version_id)

    @deprecated(
        details="get_or_create_target_dataset_version will be removed in a future version. Use the new input system instead."
    )
    def get_or_create_target_dataset_version(
        self,
        input_dataset_version: DatasetVersion,
        target_version_name: str,
    ) -> DatasetVersion:
        dataset_name = input_dataset_version.name
        dataset = self.client.get_dataset(name=dataset_name)
        try:
            output_dataset_version = dataset.create_version(version=target_version_name)
        except ResourceConflictError:
            output_dataset_version = dataset.get_version(version=target_version_name)
        return output_dataset_version
