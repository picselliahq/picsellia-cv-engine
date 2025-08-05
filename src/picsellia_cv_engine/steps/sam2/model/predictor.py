from picsellia_cv_engine import step
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.models import PicselliaPolygonPrediction
from picsellia_cv_engine.frameworks.sam2.model.model import SAM2Model
from picsellia_cv_engine.frameworks.sam2.services.predictor import SAM2ModelPredictor


@step
def predict(model: SAM2Model, dataset: CocoDataset) -> list[PicselliaPolygonPrediction]:
    """
    Inference step for generating polygon predictions from a fine-tuned SAM2 model.

    This step performs image-level inference using SAM2 on all assets in the given dataset,
    and post-processes the output into Picsellia-compatible polygon predictions.

    Args:
        model (Model): The Picsellia model instance containing trained weights.
        dataset (CocoDataset): A dataset of images to perform inference on.

    Returns:
        list[PicselliaPolygonPrediction]: A list of predictions for each asset,
        including polygons, confidence scores, and associated labels.
    """
    predictor = SAM2ModelPredictor(model=model)
    image_paths = predictor.pre_process_dataset(dataset)
    results = predictor.run_inference_on_images(image_paths)
    predictions = predictor.post_process_results(image_paths, results, dataset)

    return predictions
