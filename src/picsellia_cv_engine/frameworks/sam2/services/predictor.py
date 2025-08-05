import os

import numpy as np
import supervision as sv
from PIL import Image

from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.models.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from picsellia_cv_engine.core.services.model.predictor.model_predictor import (
    ModelPredictor,
)
from picsellia_cv_engine.frameworks.sam2.model.model import SAM2Model


class SAM2ModelPredictor(ModelPredictor):
    """
    Predictor class for generating segmentation predictions using a fine-tuned SAM2 model.

    This class wraps loading the model, preprocessing the dataset, running inference,
    and formatting results into Picsellia-compatible predictions.
    """

    def __init__(self, model: SAM2Model):
        super().__init__(model=model)
        self.model = model

    def pre_process_dataset(self, dataset: CocoDataset) -> list[str]:
        """
        Collects image file paths from the dataset.

        Args:
            dataset (CocoDataset): Dataset object containing image directory.

        Returns:
            list[str]: List of full paths to image files.
        """
        return [
            os.path.join(dataset.images_dir, f)
            for f in os.listdir(dataset.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def run_inference_on_images(self, image_paths: list[str]) -> list[list[dict]]:
        """
        Runs segmentation mask inference using SAM2 on a batch of image paths.

        Args:
            image_paths (list[str]): List of image file paths.

        Returns:
            list[list[dict]]: A list of prediction dictionaries per image.
        """
        return [
            self.model.loaded_generator.generate(np.array(Image.open(p).convert("RGB")))
            for p in image_paths
        ]

    def post_process_results(
        self,
        image_paths: list[str],
        results: list[list[dict]],
        dataset: CocoDataset,
    ) -> list[PicselliaPolygonPrediction]:
        """
        Converts raw SAM2 masks into Picsellia polygon predictions.

        Args:
            image_paths (list[str]): List of image paths used for inference.
            results (list[list[dict]]): Corresponding list of mask outputs from SAM2.
            dataset (CocoDataset): Dataset object to resolve asset and label metadata.

        Returns:
            list[PicselliaPolygonPrediction]: List of predictions ready to be logged to Picsellia.
        """
        predictions = []
        label_id = list(dataset.labelmap.values())[0]

        for img_path, mask_result in zip(image_paths, results):
            filename = os.path.basename(img_path)
            asset_id = os.path.splitext(filename)[0]
            asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

            polygons, labels, confidences = [], [], []

            for mask_dict in mask_result:
                mask = mask_dict.get("segmentation")
                if mask is None:
                    continue

                poly_list = sv.mask_to_polygons(mask.astype(np.uint8))
                for poly in poly_list:
                    if len(poly) == 0:
                        continue
                    polygons.append(
                        PicselliaPolygon([[int(x), int(y)] for x, y in poly])
                    )
                    labels.append(PicselliaLabel(label_id))
                    confidences.append(PicselliaConfidence(1.0))

            if polygons:
                predictions.append(
                    PicselliaPolygonPrediction(
                        asset=asset,
                        polygons=polygons,
                        labels=labels,
                        confidences=confidences,
                    )
                )

        return predictions
