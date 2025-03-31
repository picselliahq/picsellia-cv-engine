import logging
import os
from collections import defaultdict
from typing import Any

from picsellia import Datalake
from picsellia.types.enums import TagTarget

from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.services.base.data.dataset.uploader import DatasetUploader

logger = logging.getLogger("picsellia")


class ClassificationDatasetUploader(DatasetUploader):
    def __init__(
        self,
        dataset: CocoDataset,
    ):
        super().__init__(dataset.dataset_version)
        self.dataset = dataset

    def upload_images(
        self,
        datalake: Datalake,
        data_tags: list[str] | None = None,
        batch_size: int = 10000,
    ) -> None:
        """
        Uploads images extracted from the COCO file.
        Iterates over images grouped by category and uploads existing ones in batches.
        """
        coco_data = self.dataset.load_coco_file_data()
        images_by_category = self._process_coco_data(coco_data)

        for category_name, image_paths in images_by_category.items():
            existing_paths = [path for path in image_paths if os.path.exists(path)]
            if existing_paths:
                self._add_images_to_dataset_version_in_batches(
                    datalake=datalake,
                    images_to_upload=existing_paths,
                    data_tags=data_tags,
                    asset_tags=[category_name],
                    batch_size=batch_size,
                )

            missing_paths = set(image_paths) - set(existing_paths)
            if missing_paths:
                logger.warning(
                    f"The following image files were not found for category '{category_name}': {missing_paths}"
                )

    def upload_annotations(self) -> None:
        """
        Uploads annotations by converting tags to classification annotations.
        """
        conversion_job = self.dataset.dataset_version.convert_tags_to_classification(
            tag_type=TagTarget.ASSET,
            tags=self.dataset.dataset_version.list_asset_tags(),
        )
        conversion_job.wait_for_done()

    def upload_dataset(
        self,
        datalake: Datalake,
        data_tags: list[str] | None = None,
        batch_size: int = 10000,
    ) -> None:
        """
        Fully uploads the dataset by calling both image and annotation uploads.
        """
        self.upload_images(datalake, data_tags, batch_size)
        self.upload_annotations()

    def _process_coco_data(self, coco_data: dict[str, Any]) -> dict[str, list[str]]:
        """
        Process COCO data to group image paths by category name.
        """
        if not self.dataset.images_dir:
            raise ValueError("No images directory found in the dataset.")
        category_id_to_name = {
            cat["id"]: cat["name"] for cat in coco_data["categories"]
        }
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}

        images_by_category = defaultdict(list)
        for annotation in coco_data["annotations"]:
            image_info = image_id_to_info[annotation["image_id"]]
            category_name = category_id_to_name[annotation["category_id"]]
            image_path = os.path.join(self.dataset.images_dir, image_info["file_name"])

            images_by_category[category_name].append(image_path)

        return dict(images_by_category)
