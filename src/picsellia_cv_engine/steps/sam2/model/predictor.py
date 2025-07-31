import torch

from picsellia_cv_engine import step
from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.models import PicselliaPolygonPrediction
from picsellia_cv_engine.frameworks.sam2.services.predictor import SAM2ModelPredictor


@step
def predict(model: Model, dataset: CocoDataset) -> list[PicselliaPolygonPrediction]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = SAM2ModelPredictor(model=model, device=device)
    image_paths = predictor.pre_process_dataset(dataset)
    results = predictor.run_inference_on_images(image_paths)
    predictions = predictor.post_process_results(image_paths, results, dataset)

    return predictions
