import logging
import os

import pandas as pd
from picsellia import Experiment
from picsellia.sdk.asset import Asset, MultiAsset
from picsellia.types.enums import AddEvaluationType, InferenceType
from pycocotools.coco import COCO

from picsellia_cv_engine.models.model import (
    PicselliaClassificationPrediction,
    PicselliaOCRPrediction,
    PicselliaPolygonPrediction,
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.models.steps.model.evaluator.utils.coco_converter import (
    create_coco_files_from_experiment,
)
from picsellia_cv_engine.models.steps.model.evaluator.utils.coco_utils import (
    evaluate_category,
    fix_coco_ids,
    match_image_ids,
)
from picsellia_cv_engine.models.steps.model.evaluator.utils.log_metrics import (
    upload_metrics_to_picsellia,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles the evaluation process of various prediction types for an experiment in Picsellia.

    This class processes different types of predictions (OCR, rectangles, classifications, and polygons)
    and adds them as evaluations to an experiment. It supports logging and managing the evaluation results
    based on the type of prediction.

    Attributes:
        experiment (Experiment): The Picsellia experiment to which evaluations will be added.
    """

    def __init__(self, experiment: Experiment, inference_type: InferenceType) -> None:
        """
        Initializes the ModelEvaluator with the given experiment.

        Args:
            experiment (Experiment): The Picsellia experiment object where the evaluations will be logged.
        """
        self.experiment = experiment
        self.inference_type = inference_type

    def evaluate(
        self,
        picsellia_predictions: (
            list[PicselliaClassificationPrediction]
            | list[PicselliaRectanglePrediction]
            | list[PicselliaPolygonPrediction]
            | list[PicselliaOCRPrediction]
        ),
    ) -> None:
        """
        Evaluates a list of predictions and adds them to the experiment.

        This method processes a list of predictions and delegates the task to `add_evaluation`
        for each prediction based on its type.

        Args:
            picsellia_predictions (Union[List[PicselliaClassificationPrediction], List[PicselliaRectanglePrediction],
                List[PicselliaPolygonPrediction], List[PicselliaOCRPrediction]]):
                A list of Picsellia predictions, which can include classification, rectangle, polygon, or OCR predictions.
        """
        for prediction in picsellia_predictions:
            self.add_evaluation(prediction)
        self.experiment.compute_evaluations_metrics(inference_type=self.inference_type)

    def add_evaluation(
        self,
        evaluation: (
            PicselliaClassificationPrediction
            | PicselliaRectanglePrediction
            | PicselliaPolygonPrediction
            | PicselliaOCRPrediction
        ),
    ) -> None:
        """
        Adds a single evaluation to the experiment based on the prediction type.

        This method identifies the type of prediction and adds the corresponding evaluation
        to the Picsellia experiment. It handles OCR, rectangle, classification, and polygon
        predictions separately and logs the evaluation details.

        Args:
            evaluation (Union[PicselliaClassificationPrediction, PicselliaRectanglePrediction,
                PicselliaPolygonPrediction, PicselliaOCRPrediction]):
                A single prediction instance, which can be a classification, rectangle, polygon, or OCR prediction.

        Raises:
            TypeError: If the prediction type is not supported.
        """
        asset = evaluation.asset

        if isinstance(evaluation, PicselliaOCRPrediction):
            rectangles = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                )
                for rectangle, label, conf in zip(
                    evaluation.boxes,
                    evaluation.labels,
                    evaluation.confidences,
                    strict=False,
                )
            ]
            rectangles_with_text = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                    text.value,
                )
                for rectangle, label, conf, text in zip(
                    evaluation.boxes,
                    evaluation.labels,
                    evaluation.confidences,
                    evaluation.texts,
                    strict=False,
                )
            ]
            logger.info(
                f"Adding evaluation for asset {asset.filename} with rectangles {rectangles_with_text}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, rectangles=rectangles
            )

        elif isinstance(evaluation, PicselliaRectanglePrediction):
            rectangles = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                )
                for rectangle, label, conf in zip(
                    evaluation.boxes,
                    evaluation.labels,
                    evaluation.confidences,
                    strict=False,
                )
            ]
            logger.info(
                f"Adding evaluation for asset {asset.filename} with rectangles {rectangles}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, rectangles=rectangles
            )

        elif isinstance(evaluation, PicselliaClassificationPrediction):
            classifications = [(evaluation.label.value, evaluation.confidence.value)]
            self.experiment.add_evaluation(
                asset,
                add_type=AddEvaluationType.REPLACE,
                classifications=classifications,
            )

        elif isinstance(evaluation, PicselliaPolygonPrediction):
            polygons = [
                (polygon.value, label.value, conf.value)
                for polygon, label, conf in zip(
                    evaluation.polygons,
                    evaluation.labels,
                    evaluation.confidences,
                    strict=False,
                )
            ]
            if not polygons:
                logger.info(
                    f"Adding an empty evaluation for asset {asset.filename} (no polygons found)."
                )
                self.experiment.add_evaluation(
                    asset, add_type=AddEvaluationType.REPLACE, polygons=[]
                )
            else:
                logger.info(
                    f"Adding evaluation for asset {asset.filename} with polygons {polygons}"
                )
                self.experiment.add_evaluation(
                    asset, add_type=AddEvaluationType.REPLACE, polygons=polygons
                )

        else:
            raise TypeError("Unsupported prediction type")

    def compute_coco_metrics(
        self, experiment: Experiment, assets: list[Asset] | MultiAsset, output_dir: str
    ) -> None:
        """
        Computes COCO metrics for the given experiment and assets and saves the results to a CSV file.

        Args:
            experiment (Experiment): The Picsellia experiment to which evaluations will be added.
            assets (list[Asset] | MultiAsset): The assets to be evaluated.
            output_dir (str): The directory where the evaluation results will be saved.
        """

        os.makedirs(output_dir, exist_ok=True)
        gt_coco_path = os.path.join(output_dir, "gt.json")
        pred_coco_path = os.path.join(output_dir, "pred.json")
        output_path = os.path.join(output_dir, "output.csv")

        create_coco_files_from_experiment(
            experiment=experiment,
            assets=assets,
            gt_coco_path=gt_coco_path,
            pred_coco_path=pred_coco_path,
            inference_type=self.inference_type,
        )

        gt_path_fixed = fix_coco_ids(gt_coco_path)
        pred_path_fixed = fix_coco_ids(pred_coco_path)
        matched_prediction_file = pred_path_fixed.replace(".json", "_matched.json")
        match_image_ids(gt_path_fixed, pred_path_fixed, matched_prediction_file)
        coco_gt = COCO(gt_path_fixed)
        coco_pred = COCO(matched_prediction_file)
        categories = {
            cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
        }
        results = [
            evaluate_category(
                coco_gt=coco_gt,
                coco_pred=coco_pred,
                cat_id=cat_id,
                cat_name=cat_name,
                inference_type=self.inference_type,
            )
            for cat_id, cat_name in categories.items()
        ]
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

        upload_metrics_to_picsellia(
            experiment=experiment,
            csv_path=output_path,
        )
