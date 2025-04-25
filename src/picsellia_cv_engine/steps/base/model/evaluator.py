from picsellia import Asset
from picsellia.sdk.asset import MultiAsset
from picsellia.types.enums import InferenceType

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.models.base.picsellia_prediction import (
    PicselliaClassificationPrediction,
    PicselliaOCRPrediction,
    PicselliaPolygonPrediction,
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.services.base.model.evaluator.model_evaluator import (
    ModelEvaluator,
)


@step
def evaluate_model(
    picsellia_predictions: (
        list[PicselliaClassificationPrediction]
        | list[PicselliaRectanglePrediction]
        | list[PicselliaPolygonPrediction]
        | list[PicselliaOCRPrediction]
    ),
    inference_type: InferenceType,
    assets: list[Asset] | MultiAsset,
    output_dir: str,
) -> None:
    """
    Generic evaluation step using Picsellia's ModelEvaluator.

    Args:
        picsellia_predictions: A list of PicselliaPrediction objects (classification, detection, etc.)
        inference_type: Type of inference task (classification, detection, segmentation, OCR).
        assets: The dataset assets to compare predictions against.
        output_dir: Directory to write evaluation results.
    """
    context = Pipeline.get_active_context()
    return evaluate_model_impl(
        context=context,
        picsellia_predictions=picsellia_predictions,
        inference_type=inference_type,
        assets=assets,
        output_dir=output_dir,
    )


def evaluate_model_impl(
    context,
    picsellia_predictions: (
        list[PicselliaClassificationPrediction]
        | list[PicselliaRectanglePrediction]
        | list[PicselliaPolygonPrediction]
        | list[PicselliaOCRPrediction]
    ),
    inference_type: InferenceType,
    assets: list[Asset] | MultiAsset,
    output_dir: str,
    training_labelmap: dict[str, str] | None = None,
) -> None:
    evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=inference_type
    )
    evaluator.evaluate(picsellia_predictions=picsellia_predictions)

    if inference_type == InferenceType.CLASSIFICATION:
        evaluator.compute_classification_metrics(
            assets=assets, output_dir=output_dir, training_labelmap=training_labelmap
        )
    elif inference_type in (InferenceType.OBJECT_DETECTION, InferenceType.SEGMENTATION):
        evaluator.compute_coco_metrics(
            assets=assets, output_dir=output_dir, training_labelmap=training_labelmap
        )
    else:
        raise ValueError(f"Unsupported model type: {inference_type}")
