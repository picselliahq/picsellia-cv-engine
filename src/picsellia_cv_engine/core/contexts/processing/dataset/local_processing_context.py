import os
from typing import Any

from picsellia import DatasetVersion, ModelVersion
from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.contexts import (
    PicselliaContext,
)


class LocalProcessingContext(PicselliaContext):
    """
    This class is used to test a processing pipeline without a real job execution on Picsellia (without giving a real job ID).
    """

    def __init__(
        self,
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        job_id: str | None = None,
        job_type: ProcessingType | None = None,
        input_dataset_version_id: str | None = None,
        output_dataset_version_id: str | None = None,
        output_dataset_version_name: str | None = None,
        use_id: bool | None = True,
        download_annotations: bool | None = True,
        model_version_id: str | None = None,
        processing_parameters=None,
        working_dir: str | None = None,
    ):
        # Initialize the Picsellia client from the base class
        super().__init__(
            api_token=api_token,
            host=host,
            organization_id=organization_id,
            organization_name=organization_name,
            working_dir=working_dir,
        )
        self.job_id = job_id
        self.job_type = job_type
        self.input_dataset_version_id = input_dataset_version_id
        self.output_dataset_version_id = output_dataset_version_id
        self.model_version_id = model_version_id
        if self.input_dataset_version_id:
            self.input_dataset_version = self.get_dataset_version(
                self.input_dataset_version_id
            )
        if self.output_dataset_version_id:
            self.output_dataset_version = self.get_dataset_version(
                self.output_dataset_version_id
            )
        elif output_dataset_version_name:
            self.output_dataset_version = self.client.get_dataset_by_id(
                self.input_dataset_version.origin_id
            ).create_version(version=output_dataset_version_name)
            self.output_dataset_version_id = self.output_dataset_version.id
        if self.model_version_id:
            self.model_version = self.get_model_version()
        self.processing_parameters = processing_parameters
        self.use_id = use_id
        self.download_annotations = download_annotations

    @property
    def working_dir(self) -> str:
        if self._working_dir_override:
            return self._working_dir_override
        return os.path.join(os.getcwd(), f"job_{self.job_id}")

    def get_dataset_version(self, dataset_version_id) -> DatasetVersion:
        """
        Fetches the dataset version from Picsellia using the input dataset version ID.

        The DatasetVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a dataset.

        Returns:
            The dataset version fetched from Picsellia.
        """
        return self.client.get_dataset_version_by_id(dataset_version_id)

    def get_model_version(self) -> ModelVersion:
        """
        Fetches the models version from Picsellia using the models version ID.

        The ModelVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a models.

        Returns:
            The models version fetched from Picsellia.
        """
        return self.client.get_model_version_by_id(self.model_version_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_type": self.job_type,
                "input_dataset_version_id": self.input_dataset_version_id,
                "output_dataset_version_id": self.output_dataset_version_id,
                "model_version_id": self.model_version_id,
                "use_id": self.use_id,
            },
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
        }
