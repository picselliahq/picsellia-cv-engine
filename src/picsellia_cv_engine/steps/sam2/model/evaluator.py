import os

import torch

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import (
    LocalTrainingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from picsellia_cv_engine.frameworks.sam2.model.model import SAM2Model
from picsellia_cv_engine.frameworks.sam2.services.predictor import SAM2ModelPredictor


@step
def evaluate(model: SAM2Model, dataset: CocoDataset):
    """
    Evaluation step for the SAM2 model using a given COCO dataset.

    This step performs full inference on all dataset images using the trained SAM2 model,
    then post-processes the output into Picsellia-compatible predictions and evaluates them.

    Args:
        model (Model): The trained Picsellia SAM2 model to be evaluated.
        dataset (CocoDataset): The COCO-format dataset used for evaluation.

    Returns:
        None
    """
    context: PicselliaTrainingContext | LocalTrainingContext = (
        Pipeline.get_active_context()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model.trained_weights_path:
        raise FileNotFoundError("No trained weights path found in model.")

    if not model.config_path:
        raise FileNotFoundError("No configuration path found in model.")

    loaded_model, loaded_generator = model.load_weights(
        weights_path=model.trained_weights_path,
        config_path=model.config_path,
        device=device,
    )
    model.set_loaded_model(loaded_model)
    model.set_loaded_generator(loaded_generator)

    predictor = SAM2ModelPredictor(model=model)
    image_paths = predictor.pre_process_dataset(dataset)
    results = predictor.run_inference_on_images(image_paths)
    predictions = predictor.post_process_results(image_paths, results, dataset)

    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=model.model_version.type,
        assets=dataset.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=context.experiment.get_log("labelmap").data,
    )
