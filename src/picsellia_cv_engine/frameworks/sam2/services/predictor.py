import os
from typing import Optional

import numpy as np
from PIL import Image

from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.services.model.predictor.model_predictor import (
    ModelPredictor,
)
from picsellia_cv_engine.core.services.utils.annotations import mask_to_polygons
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

    def pre_process_dataset(self, dataset: CocoDataset) -> list[np.ndarray]:
        """
        Collects image file paths from the dataset.

        Args:
            dataset (CocoDataset): Dataset object containing image directory.

        Returns:
            list[str]: List of full paths to image files.
        """
        images = []
        for f in os.listdir(dataset.images_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(dataset.images_dir, f)
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)
                images.append(img_np)
        return images

    def preprocess_images(self, image_list: list[np.ndarray]):
        self.model.loaded_predictor.set_image_batch(image_list=image_list)

    def preprocess(self, image: np.ndarray):
        self.model.loaded_predictor.set_image(image=image)

    def run_inference(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ):
        masks, ious, _ = self.model.loaded_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

        mask_dicts = [
            {"segmentation": masks[i], "score": float(ious[i])}
            for i in range(len(masks))
        ]
        return mask_dicts

    def post_process(self, results: list[dict]):
        polygons = []
        for mask_dict in results:
            mask = mask_dict.get("segmentation")
            if mask is None:
                continue

            poly_list = mask_to_polygons(mask.astype(np.uint8))
            for poly in poly_list:
                if len(poly) == 0:
                    continue
                polygons.append([[int(x), int(y)] for x, y in poly])
        return polygons
