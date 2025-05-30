from abc import abstractmethod
from typing import Optional

from picsellia import Client, Data, Datalake, DatasetVersion
from picsellia.services.error_manager import ErrorManager
from picsellia.types.enums import InferenceType


class DatasetVersionCreationProcessing:
    """
    Handles the processing of creating a dataset version.

    This class offers all the necessary methods to handle a processing of type DatasetVersionCreation of Picsellia.
    It allows to upload images to the datalake, add them to a dataset version, and update the dataset version with
    the necessary information.

    Attributes:
        client (Client): The Picsellia client to use for the processing.
        datalake(Datalake): The Datalake to use for the processing.
        output_dataset_version (DatasetVersion): The dataset version to create.
    """

    def __init__(
        self,
        client: Client,
        datalake: Datalake,
        output_dataset_version: DatasetVersion,
    ):
        self.client = client
        self.output_dataset_version = output_dataset_version
        self.datalake = datalake

    def update_output_dataset_version_description(self, description: str) -> None:
        """
        Updates the description of the output dataset version.

        Args:
            description (str): The new description to set for the dataset version.

        """
        self.output_dataset_version.update(description=description)

    def update_output_dataset_version_inference_type(
        self, inference_type: InferenceType
    ) -> None:
        """
        Updates the inference type of the output dataset version.

        Args:
            inference_type (InferenceType): The new inference type to set for the dataset version.

        """
        self.output_dataset_version.update(type=inference_type)

    def _upload_data_with_error_manager(
        self, images_to_upload: list[str], images_tags: Optional[list[str]] = None
    ) -> tuple[list[Data], list[str]]:
        """
        Uploads data to the datalake using an error manager. This method allows to handle errors during the upload process.
        It will retry to upload the data that failed to upload.

        Args:
            images_to_upload (list[str]): The list of image file paths to upload.
            images_tags (Optional[list[str]]): The list of tags to associate with the images.

        Returns:
            - list[Data]: The list of uploaded data.
            - list[str]: The list of file paths that failed to upload.
        """
        error_manager = ErrorManager()
        data = self.datalake.upload_data(
            filepaths=images_to_upload, tags=images_tags, error_manager=error_manager
        )

        if isinstance(data, Data):
            uploaded_data = [data]
        else:
            uploaded_data = list(data)

        error_paths = [error.path for error in error_manager.errors]
        return uploaded_data, error_paths

    def _upload_images_to_datalake(
        self,
        images_to_upload: list[str],
        images_tags: Optional[list[str]] = None,
        max_retries: int = 5,
    ) -> list[Data]:
        """
        Uploads images to the datalake. This method allows to handle errors during the upload process.

        Args:
            images_to_upload (list[str]): The list of image file paths to upload.
            images_tags (Optional[list[str]]): The list of tags to associate with the images.
            max_retries (int): The maximum number of retries to upload the images.

        Returns:

        """
        all_uploaded_data = []
        uploaded_data, error_paths = self._upload_data_with_error_manager(
            images_to_upload=images_to_upload, images_tags=images_tags
        )
        all_uploaded_data.extend(uploaded_data)
        retry_count = 0
        while error_paths and retry_count < max_retries:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                images_to_upload=error_paths, images_tags=images_tags
            )
            all_uploaded_data.extend(uploaded_data)
            retry_count += 1
        if error_paths:
            raise Exception(
                f"Failed to upload the following images: {error_paths} after {max_retries} retries."
            )
        return all_uploaded_data

    def _add_images_to_dataset_version(
        self,
        images_to_upload: list[str],
        images_tags: Optional[list[str]] = None,
        max_retries: int = 5,
    ) -> None:
        """
        Adds images to the dataset version.

        Args:
            images_to_upload (list[str]): The list of image file paths to upload.
            images_tags (Optional[list[str]]): The list of tags to associate with the images.
            max_retries (int): The maximum number of retries to upload the images.

        """
        data = self._upload_images_to_datalake(
            images_to_upload=images_to_upload,
            images_tags=images_tags,
            max_retries=max_retries,
        )
        self.output_dataset_version.add_data(data=data)

    def _add_coco_annotations_to_dataset_version(self, annotation_path: str):
        """
        Adds COCO annotations to the dataset version.

        Args:
            annotation_path (str): The path to the COCO annotations file.

        """
        self.output_dataset_version.import_annotations_coco_file(
            file_path=annotation_path
        )

    @abstractmethod
    def process(self) -> None:
        """
        Processes the dataset version creation.
        """
        pass
