from typing import Any, Generic, TypeVar
from uuid import UUID

import orjson
from deprecation import deprecated
from picsellia import Client, Datalake, Job, ModelVersion
from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.contexts.processing.common.local_picsellia_context import (
    PicselliaLocalProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


def create_processing(
    client: Client,
    name: str,
    type: str | ProcessingType,
    default_cpu: int,
    default_gpu: int,
    default_parameters: dict,
    docker_image: str,
    docker_tag: str,
    docker_flags: list[str] | None = None,
) -> str:
    """
    Create a processing configuration in Picsellia.

    Returns:
        str: ID of the created processing.
    """
    payload = {
        "name": name,
        "type": type,
        "default_cpu": default_cpu,
        "default_gpu": default_gpu,
        "default_parameters": default_parameters,
        "docker_image": docker_image,
        "docker_tag": docker_tag,
        "docker_flags": docker_flags,
    }
    r = client.connexion.post(
        f"/sdk/organization/{client.id}/processings", data=orjson.dumps(payload)
    ).json()
    return r["id"]


def get_processing(client: Client, name: str) -> str:
    """
    Get the ID of a processing by name.

    Returns:
        str: ID of the found processing.
    """
    r = client.connexion.get(
        f"/sdk/organization/{client.id}/processings", params={"name": name}
    ).json()
    return r["items"][0]["id"]


def launch_processing(
    client: Client,
    datalake: Datalake,
    data_ids: list[UUID],
    model_version_id: str,
    processing_id: str,
    parameters: dict,
    cpu: int,
    gpu: int,
    target_datalake_name: str | None = None,
) -> Job:
    """
    Launch a processing job on a datalake.

    Returns:
        Job: The launched job object.
    """
    payload = {
        "processing_id": processing_id,
        "parameters": parameters,
        "cpu": cpu,
        "gpu": gpu,
        "model_version_id": model_version_id,
        "data_ids": data_ids,
    }

    if target_datalake_name:
        payload["target_datalake_name"] = target_datalake_name

    r = client.connexion.post(
        f"/api/datalake/{datalake.id}/processing/launch",
        data=orjson.dumps(payload),
    ).json()
    return Job(client.connexion, r, version=2)


class LocalDatalakeProcessingContext(
    PicselliaLocalProcessingContext, Generic[TParameters]
):
    """
    Context for local testing of processing jobs without real job execution on Picsellia.
    """

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
        offset: int = 0,
        limit: int = 100,
        use_id: bool | None = True,
        working_dir: str | None = None,
    ):
        """
        Initialize the local datalake processing context.

        Raises:
            ValueError: If the input datalake ID is missing or invalid.
        """
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
            use_id=use_id,
            working_dir=working_dir,
        )

        self.target = self.client.get_datalake(id=self.target_id)

        self.offset = offset
        self.limit = limit

        self.data_ids = None
        if self.limit is not None and self.offset is not None:
            self.data_ids = self.get_data_ids(offset=self.offset, limit=self.limit)

    def _load_legacy_inputs(self, **kwargs) -> None:
        self._model_version_id = self.inputs.get("model_version_id")
        self._input_datalake_id = self.target_id
        self._output_datalake_id = self.inputs.get("output_datalake_id")

        self.input_datalake = self.get_datalake(self._input_datalake_id)
        self.output_datalake = (
            self.get_datalake(self._output_datalake_id)
            if self._output_datalake_id
            else None
        )

        self.model_version = (
            self.get_model_version() if self._model_version_id else None
        )

    @deprecated(
        details="get_datalake will be removed in a future version. Use the new input system instead."
    )
    def get_datalake(self, datalake_id: str) -> Datalake:
        return self.client.get_datalake(id=datalake_id)

    @deprecated(
        details="get_model_version will be removed in a future version. Use the new input system instead."
    )
    def get_model_version(self) -> ModelVersion:
        return self.client.get_model_version_by_id(self._model_version_id)

    def get_data_ids(self, offset: int, limit: int) -> list[UUID]:
        """List data IDs from a datalake with offset and limit."""
        if not self.target or offset is None or limit is None:
            raise ValueError("Datalake, offset and limit must be provided")
        return self.target.list_data(offset=offset, limit=limit).ids

    def to_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary for logging or serialization."""
        base = super().to_dict()
        base.update(
            {
                "job_type": self.job_type,
                "input_datalake_id": self._input_datalake_id,
                "output_datalake_id": self._output_datalake_id,
                "model_version_id": self._model_version_id,
                "offset": self.offset,
                "limit": self.limit,
                "use_id": self.use_id,
            }
        )
        return base
