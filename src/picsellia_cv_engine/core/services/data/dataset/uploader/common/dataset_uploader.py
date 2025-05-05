import json
import os

from picsellia import Data, Datalake, DatasetVersion
from picsellia.types.enums import ImportAnnotationMode

from picsellia_cv_engine.core.services.data.dataset.uploader.common import DataUploader


class DatasetUploader(DataUploader):
    """
    Utility class to upload images and annotations to a Picsellia dataset version.
    """

    def __init__(self, dataset_version: DatasetVersion):
        """
        Initialize with a target dataset version.

        Args:
            dataset_version (DatasetVersion): The dataset version to populate.
        """
        self.dataset_version = dataset_version

    def _add_data_to_dataset_version_and_wait(
        self, data: list[Data], asset_tags: list[str] | None = None
    ):
        """
        Add data to the dataset version and wait for the job to finish.

        Args:
            data (list[Data]): Uploaded data to add.
            asset_tags (list[str] | None): Optional tags for the assets.
        """
        adding_job = self.dataset_version.add_data(data=data, tags=asset_tags)
        adding_job.wait_for_done()

    def _add_images_to_dataset_version(
        self,
        datalake: Datalake,
        images_to_upload: list[str],
        data_tags: list[str] | None = None,
        asset_tags: list[str] | None = None,
        max_retries: int = 5,
    ) -> None:
        """
        Upload images and add them to the dataset version.

        Args:
            datalake (Datalake): Target datalake.
            images_to_upload (list[str]): Paths of images to upload.
            data_tags (list[str] | None): Tags for uploaded data.
            asset_tags (list[str] | None): Tags for dataset assets.
            max_retries (int): Max retry attempts on failure.
        """
        data = self._upload_images_to_datalake(
            datalake=datalake,
            images_to_upload=images_to_upload,
            data_tags=data_tags,
            max_retries=max_retries,
        )
        self._add_data_to_dataset_version_and_wait(data=data, asset_tags=asset_tags)

    def _add_images_to_dataset_version_in_batches(
        self,
        datalake: Datalake,
        images_to_upload: list[str],
        data_tags: list[str] | None = None,
        asset_tags: list[str] | None = None,
        batch_size: int = 10000,
        max_retries: int = 5,
    ) -> None:
        """
        Upload images and add them to the dataset version in batches.

        Args:
            datalake (Datalake): Target datalake.
            images_to_upload (list[str]): Paths of images to upload.
            data_tags (list[str] | None): Tags for uploaded data.
            asset_tags (list[str] | None): Tags for dataset assets.
            batch_size (int): Number of assets per add job.
            max_retries (int): Max retry attempts on failure.
        """
        uploaded_data = self._upload_images_to_datalake(
            datalake=datalake,
            images_to_upload=images_to_upload,
            data_tags=data_tags,
            max_retries=max_retries,
        )

        batches = [
            uploaded_data[i : i + batch_size]
            for i in range(0, len(uploaded_data), batch_size)
        ]

        jobs = [
            self.dataset_version.add_data(data=batch, tags=asset_tags)
            for batch in batches
        ]

        for job in jobs:
            job.wait_for_done()

    def _add_coco_annotations_to_dataset_version(
        self,
        annotation_path: str,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
        replace_annotations: bool = False,
    ):
        """
        Import a full COCO annotation file into the dataset version.

        Args:
            annotation_path (str): Path to COCO JSON file.
            use_id (bool): Match using asset ID.
            fail_on_asset_not_found (bool): Whether to raise if assets are missing.
            replace_annotations (bool): Whether to replace existing annotations.
        """
        mode = (
            ImportAnnotationMode.REPLACE
            if replace_annotations
            else ImportAnnotationMode.KEEP
        )

        self.dataset_version.import_annotations_coco_file(
            file_path=annotation_path,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
            mode=mode,
        )

    def _split_coco_annotations(
        self, coco_data: dict, batch_image_ids: list[str]
    ) -> dict:
        """
        Create a mini-COCO file with only selected image IDs.

        Args:
            coco_data (dict): Original COCO data.
            batch_image_ids (list[str]): IDs of images to include.

        Returns:
            dict: Filtered COCO data.
        """
        return {
            "images": [
                img for img in coco_data["images"] if img["id"] in batch_image_ids
            ],
            "annotations": [
                ann
                for ann in coco_data["annotations"]
                if ann["image_id"] in batch_image_ids
            ],
            "categories": coco_data["categories"],
        }

    def _add_coco_annotations_to_dataset_version_in_batches(
        self,
        annotation_path: str,
        batch_size: int = 10000,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
        replace_annotations: bool = False,
    ):
        """
        Import COCO annotations in batches to avoid request overload.

        Args:
            annotation_path (str): Path to full COCO annotation file.
            batch_size (int): Max number of images per batch.
            use_id (bool): Match using asset ID.
            fail_on_asset_not_found (bool): Whether to raise if assets are missing.
            replace_annotations (bool): Whether to replace existing annotations.
        """
        with open(annotation_path) as f:
            coco_data = json.load(f)

        image_ids = [img["id"] for img in coco_data["images"]]
        batches = [
            image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)
        ]

        temp_path = "temp_batch_annotations.json"

        for batch_ids in batches:
            batch_data = self._split_coco_annotations(coco_data, batch_ids)
            if not batch_data["images"]:
                continue

            with open(temp_path, "w") as f:
                json.dump(batch_data, f)

            self._add_coco_annotations_to_dataset_version(
                annotation_path=temp_path,
                use_id=use_id,
                fail_on_asset_not_found=fail_on_asset_not_found,
                replace_annotations=replace_annotations,
            )

        os.remove(temp_path)
