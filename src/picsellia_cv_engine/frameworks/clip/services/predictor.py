import os
from dataclasses import dataclass

import torch
from picsellia import Asset
from PIL import Image

from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.services.model.predictor.model_predictor import (
    ModelPredictor,
)
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel


@dataclass
class PicselliaCLIPEmbeddingPrediction:
    """Predictions contenant les embeddings CLIP image + texte"""

    asset: Asset
    image_embedding: list[float]
    text_embedding: list[float]


class CLIPModelPredictor(ModelPredictor):
    def __init__(self, model: CLIPModel, device: str):
        super().__init__(model=model)
        self.model = model
        self.device = device

    def embed_image(self, image_path: str) -> list[float]:
        """Encode an image into a CLIP embedding."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.model.loaded_processor(images=image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            image_emb = self.model.loaded_model.get_image_features(**inputs)

        return image_emb[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Encode a text string into a CLIP embedding."""
        inputs = self.model.loaded_processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            text_emb = self.model.loaded_model.get_text_features(**inputs)

        return text_emb[0].cpu().tolist()

    def run_image_inference_on_batches(
        self, image_batches: list[list[str]]
    ) -> list[list[dict]]:
        results = []
        for batch in image_batches:
            batch_results = []
            for image_path in batch:
                embedding = self.embed_image(image_path)
                batch_results.append({"image_embedding": embedding})
            results.append(batch_results)
        return results

    def run_inference_on_batches(
        self, image_text_batches: list[list[tuple[str, str]]]
    ) -> list[list[dict]]:
        results = []
        for batch in image_text_batches:
            batch_results = []
            for image_path, text in batch:
                result = {
                    "image_embedding": self.embed_image(image_path),
                    "text_embedding": self.embed_text(text),
                }
                batch_results.append(result)
            results.append(batch_results)
        return results

    def post_process_batches(
        self,
        image_text_batches: list[list[tuple[str, str]]],
        batch_results: list[list[dict]],
        dataset: TBaseDataset,
    ) -> list[PicselliaCLIPEmbeddingPrediction]:
        all_predictions = []

        for image_texts, results in zip(image_text_batches, batch_results):
            for (image_path, _), result in zip(image_texts, results):
                asset_id = os.path.splitext(os.path.basename(image_path))[0]
                asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

                prediction = PicselliaCLIPEmbeddingPrediction(
                    asset=asset,
                    image_embedding=result["image_embedding"],
                    text_embedding=result["text_embedding"],
                )
                all_predictions.append(prediction)

        return all_predictions

    def post_process_image_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[list[dict]],
        dataset: TBaseDataset,
    ) -> list[PicselliaCLIPEmbeddingPrediction]:
        all_predictions = []
        for batch, results in zip(image_batches, batch_results):
            for image_path, result in zip(batch, results):
                asset_id = os.path.splitext(os.path.basename(image_path))[0]
                asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

                prediction = PicselliaCLIPEmbeddingPrediction(
                    asset=asset,
                    image_embedding=result["image_embedding"],
                    text_embedding=[],  # vide pour l’instant
                )
                all_predictions.append(prediction)
        return all_predictions
