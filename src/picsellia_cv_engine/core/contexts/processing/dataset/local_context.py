from typing import Any, Generic, TypeVar

from deprecation import deprecated
from picsellia import DatasetVersion, ModelVersion
from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.contexts.processing.common.local_picsellia_context import (
    PicselliaLocalProcessingContext,
)
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class LocalDatasetProcessingContext(
    PicselliaLocalProcessingContext, Generic[TParameters]
):
    """
    Local context for testing a processing pipeline without executing a real job on Picsellia.
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
        input_dataset_version_id: str | None = None,
        output_dataset_version_id: str | None = None,
        target_version_name: str | None = None,
        target_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        use_id: bool | None = True,
        download_annotations: bool | None = True,
        model_version_id: str | None = None,
        working_dir: str | None = None,
    ):
        """
        Initialize the local processing context.

        Can create or retrieve dataset versions and model versions as needed.

        Raises:
            ValueError: If required data is missing (e.g., input dataset version).
        """
        self.download_annotations = download_annotations
        self.job_type = job_type  # TODO: remove this after full deprecation of legacy processing jobs
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
            model_version_id=model_version_id,
            input_dataset_version_id=input_dataset_version_id,
            output_dataset_version_id=output_dataset_version_id,
            target_version_name=target_version_name,
        )

        self.target = self.client.get_dataset_version_by_id(id=self.target_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to a dictionary for logging or serialization."""
        base = super().to_dict()
        base.update(
            {
                "model_version_id": self.model_version_id,
                "input_dataset_version_id": self.input_dataset_version_id,
                "output_dataset_version_id": str(self.output_dataset_version.id),
            }
        )
        return base

    @property
    @deprecated(
        details="input_dataset_version_id will be removed in a future version. Use the new input system instead."
    )
    def input_dataset_version_id(self) -> str:
        if not self._input_dataset_version_id:
            raise ValueError("Input dataset version ID is missing.")
        return self._input_dataset_version_id

    @property
    @deprecated(
        details="target_version_name will be removed in a future version. Use the new input system instead."
    )
    def target_version_name(self) -> str:
        if not self._target_version_name:
            raise ValueError("Target (output) version name is missing.")
        return self._target_version_name

    @property
    @deprecated(
        details="model_version_id will be removed in a future version. Use the new input system instead."
    )
    def model_version_id(self) -> str | None:
        if not self._model_version_id:
            raise ValueError("Model version ID is required for pre-annotation jobs.")
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

    def _load_legacy_inputs(self, **kwargs) -> None:
        self.inputs["model_version_id"] = kwargs.get("model_version_id")
        self.inputs["input_dataset_version_id"] = kwargs.get("input_dataset_version_id")
        self.inputs["output_dataset_version_id"] = kwargs.get(
            "output_dataset_version_id"
        )
        self.inputs["target_version_name"] = kwargs.get("target_version_name")
        self._model_version_id = self.inputs.get("model_version_id")
        self._input_dataset_version_id = self.inputs.get("input_dataset_version_id")
        self._target_version_name = self.inputs.get("target_version_name")

        self.input_dataset_version = self.get_dataset_version(
            self.input_dataset_version_id
        )

        if self.target_version_name:
            self.output_dataset_version = self.get_or_create_target_dataset_version(
                input_dataset_version=self.input_dataset_version,
                target_version_name=self.target_version_name,
            )
        else:
            self.output_dataset_version = self.input_dataset_version

        if self._model_version_id:
            self.model_version = self.get_model_version()
