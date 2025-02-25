import logging
import os

from picsellia import Datalake
from picsellia.types.enums import InferenceType

from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data_upload.data_uploader import (
    DataUploader,
)

logger = logging.getLogger("picsellia")


class ObjectDetectionDatasetContextUploader(DataUploader):
    """
    Handles uploading the dataset context for object detection tasks to Picsellia.

    This class extends `DataUploader` and specifically focuses on object detection datasets.
    It uploads images to a specified datalake and, if the dataset version is correctly configured,
    uploads COCO annotations as well.
    """

    def __init__(
        self,
        dataset_context: CocoDatasetContext,
    ):
        """
        Initializes the ObjectDetectionDatasetContextUploader with a dataset context.

        Args:
            dataset_context (DatasetContext): The dataset context containing images and annotations.
        """
        super().__init__(dataset_context.dataset_version)
        self.dataset_context = dataset_context

    def upload_images(
        self,
        datalake: Datalake,
        data_tags: list[str] | None = None,
        batch_size: int = 10000,
    ) -> None:
        if self.dataset_context.images_dir:
            self._add_images_to_dataset_version_in_batches(
                datalake=datalake,
                images_to_upload=[
                    os.path.join(self.dataset_context.images_dir, image_filename)
                    for image_filename in os.listdir(self.dataset_context.images_dir)
                ],
                data_tags=data_tags,
                batch_size=batch_size,
            )

    def upload_annotations(
        self,
        batch_size: int = 10000,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
    ) -> None:
        if (
            self.dataset_context.dataset_version.type != InferenceType.NOT_CONFIGURED
            and self.dataset_context.coco_file_path
        ):
            print("Uploading annotations")
            self._add_coco_annotations_to_dataset_version_in_batches(
                annotation_path=self.dataset_context.coco_file_path,
                batch_size=batch_size,
                use_id=use_id,
                fail_on_asset_not_found=fail_on_asset_not_found,
            )
        else:
            logger.info(
                f"👉 Since the dataset's type is set to {InferenceType.NOT_CONFIGURED.name}, "
                f"no annotations will be uploaded."
            )

    def upload_dataset_context(
        self,
        datalake: Datalake,
        data_tags: list[str] | None = None,
        batch_size: int = 10000,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
    ) -> None:
        """
        Uploads the dataset context to Picsellia, including images and annotations.

        This method uploads the images from the dataset context to the datalake in batches. If the dataset version
        is configured for object detection (i.e., its type is not `NOT_CONFIGURED`), it will also upload
        COCO annotations in batches. Otherwise, it logs that no annotations will be uploaded due to the dataset type.
        """
        self.upload_images(
            datalake=datalake, data_tags=data_tags, batch_size=batch_size
        )
        self.upload_annotations(
            batch_size=batch_size,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
