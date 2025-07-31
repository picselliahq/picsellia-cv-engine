import os

import numpy as np
import supervision as sv
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.models.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)


class SAM2ModelPredictor:
    def __init__(self, model: Model, device: str = "cuda"):
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.model = build_sam2(model_cfg, model.trained_weights_path, device=device)
        self.generator = SAM2AutomaticMaskGenerator(self.model)

    def pre_process_dataset(self, dataset: CocoDataset) -> list[str]:
        return [
            os.path.join(dataset.images_dir, f)
            for f in os.listdir(dataset.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def run_inference_on_images(self, image_paths: list[str]) -> list[list[dict]]:
        return [
            self.generator.generate(np.array(Image.open(p).convert("RGB")))
            for p in image_paths
        ]

    def post_process_results(
        self,
        image_paths: list[str],
        results: list[list[dict]],
        dataset: CocoDataset,
    ) -> list[PicselliaPolygonPrediction]:
        predictions = []
        label_id = list(dataset.labelmap.values())[0]  # Use first label by default

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
                    if not poly:
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
