import json
import os

from picsellia import Data, Datalake, DatasetVersion

from picsellia_cv_engine.services.base.data.dataset.uploader import (
    DataUploader,
)


class DatasetUploader(DataUploader):
    def __init__(self, dataset_version: DatasetVersion):
        self.dataset_version = dataset_version

    def _add_data_to_dataset_version_and_wait(
        self, data: list[Data], asset_tags: list[str] | None = None
    ):
        """
        Utility function to add data to dataset version and wait for job completion.
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
        Uploads all images to the datalake first, then adds them to the dataset version in batches.

        Args:
            datalake (Datalake): Datalake to upload images to.
            images_to_upload (List[str]): List of image file paths to upload.
            data_tags (Optional[List[str]]): Tags to associate with images (default is None).
            asset_tags (Optional[List[str]]): Tags to associate with dataset version assets (default is None).
            batch_size (int): Number of images per batch for adding to dataset version.
            max_retries (int): Maximum retries for failed uploads (default is 5).
        """
        # Step 1: Upload all images to the datalake
        uploaded_data = self._upload_images_to_datalake(
            datalake=datalake,
            images_to_upload=images_to_upload,
            data_tags=data_tags,
            max_retries=max_retries,
        )

        # Step 2: Divide uploaded data into batches for adding to dataset version
        batches = [
            uploaded_data[i : i + batch_size]
            for i in range(0, len(uploaded_data), batch_size)
        ]

        # Step 3: Add data in batches to the dataset version and wait for each job to complete
        jobs = []
        for batch in batches:
            job = self.dataset_version.add_data(data=batch, tags=asset_tags)
            jobs.append(job)

        for job in jobs:
            job.wait_for_done()

    def _add_coco_annotations_to_dataset_version(
        self,
        annotation_path: str,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
    ):
        self.dataset_version.import_annotations_coco_file(
            file_path=annotation_path,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )

    def _split_coco_annotations(
        self, coco_data: dict, batch_image_ids: list[str]
    ) -> dict:
        batch_images = [
            img for img in coco_data["images"] if img["id"] in batch_image_ids
        ]
        batch_annotations = [
            ann
            for ann in coco_data["annotations"]
            if ann["image_id"] in batch_image_ids
        ]

        return {
            "images": batch_images,
            "annotations": batch_annotations,
            "categories": coco_data["categories"],
        }

    def _add_coco_annotations_to_dataset_version_in_batches(
        self,
        annotation_path: str,
        batch_size: int = 10000,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
    ):
        with open(annotation_path) as f:
            coco_data = json.load(f)

        # Extract unique image IDs
        image_ids = [image["id"] for image in coco_data["images"]]

        # Split image IDs into batches
        batches_images_ids = [
            image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)
        ]

        batch_annotation_path = "temp_batch_annotations.json"

        for batch_image_ids in batches_images_ids:
            batch_coco_data = self._split_coco_annotations(coco_data, batch_image_ids)
            if len(batch_coco_data["images"]) == 0:
                continue

            with open(batch_annotation_path, "w") as batch_file:
                json.dump(batch_coco_data, batch_file)

            self._add_coco_annotations_to_dataset_version(
                annotation_path=batch_annotation_path,
                use_id=use_id,
                fail_on_asset_not_found=fail_on_asset_not_found,
            )

        # Remove temporary file
        os.remove(batch_annotation_path)
