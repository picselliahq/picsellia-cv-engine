import torch

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from picsellia_cv_engine.frameworks.clip.services.predictor import (
    CLIPModelPredictor,
    PicselliaCLIPEmbeddingPrediction,
)


@step
def predict(
    model: CLIPModel, dataset: CocoDataset
) -> list[PicselliaCLIPEmbeddingPrediction]:
    """
    Step d'inférence CLIP sur un dataset d'images uniquement (pas de texte).
    Renvoie des embeddings image-only.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = CLIPModelPredictor(model=model, device=device)

    # 🔍 Récupère les chemins des images
    image_paths = predictor.pre_process_dataset(dataset)

    # 📦 Batch les images seules
    image_batches = predictor.prepare_batches(
        image_paths, batch_size=parameters.get("batch_size", 4)
    )

    # 🧠 Inférence sur images
    results = predictor.run_image_inference_on_batches(image_batches)

    # 📤 Post-processing en prédictions CLIP
    predictions = predictor.post_process_image_batches(
        image_batches=image_batches,
        batch_results=results,
        dataset=dataset,
    )

    return predictions
