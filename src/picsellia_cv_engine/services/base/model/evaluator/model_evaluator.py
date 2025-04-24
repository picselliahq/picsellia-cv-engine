import logging
import os

import numpy as np
import pandas as pd
from picsellia import Experiment
from picsellia.sdk.asset import Asset, MultiAsset
from picsellia.types.enums import AddEvaluationType, InferenceType
from pycocotools.coco import COCO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from picsellia_cv_engine.core.models import (
    PicselliaClassificationPrediction,
    PicselliaOCRPrediction,
    PicselliaPolygonPrediction,
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.services.base.model.evaluator.utils.coco_converter import (
    create_coco_files_from_experiment,
)
from picsellia_cv_engine.services.base.model.evaluator.utils.coco_utils import (
    evaluate_category,
    fix_coco_ids,
    match_image_ids,
)
from picsellia_cv_engine.services.base.model.evaluator.utils.compute_metrics import (
    compute_full_confusion_matrix,
)
from picsellia_cv_engine.services.base.model.logging import BaseLogger, MetricMapping

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
        self.experiment_logger = BaseLogger(
            experiment=experiment, metric_mapping=MetricMapping()
        )

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
        self, assets: list[Asset] | MultiAsset, output_dir: str
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
            experiment=self.experiment,
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

        df = pd.read_csv(output_path).round(3)

        if not df.empty:
            row_labels = df["Class"].tolist()
            columns = df.columns.drop("Class").tolist()
            matrix = df.drop(columns=["Class"]).values.tolist()

            self.experiment_logger.log_table(
                name="metrics",
                data={"data": matrix, "rows": row_labels, "columns": columns},
                phase="test",
            )

            key_name_map = {
                "Box(mAP50)": "mAP50(B)",
                "Box(mAP50-95)": "mAP50-95(B)",
                "Mask(mAP50)": "mAP50(M)",
                "Mask(mAP50-95)": "mAP50-95(M)",
            }

            for original_key, log_name in key_name_map.items():
                if original_key in df.columns:
                    mean_value = df[original_key].mean()
                    self.experiment_logger.log_value(
                        name=log_name, value=round(mean_value, 3), phase="test"
                    )

        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds())
        cat_ids = list(categories.keys())

        conf_matrix = compute_full_confusion_matrix(
            gt_annotations=gt_anns,
            pred_annotations=pred_anns,
            category_ids=cat_ids,
            iou_threshold=0.5,
        )

        label_map = dict(enumerate(categories.values()))
        label_map[len(label_map)] = "background"

        self.experiment_logger.log_confusion_matrix(
            name="confusion-matrix",
            labelmap=label_map,
            matrix=conf_matrix,
            phase="test",
        )

    def compute_classification_metrics(
        self, assets: list[Asset] | MultiAsset, output_dir: str
    ) -> None:
        """
        Compute classification metrics (accuracy, precision, recall, F1...) using COCO formatted GT and prediction files.
        """

        os.makedirs(output_dir, exist_ok=True)
        gt_coco_path = os.path.join(output_dir, "gt.json")
        pred_coco_path = os.path.join(output_dir, "pred.json")

        # Generate files
        create_coco_files_from_experiment(
            experiment=self.experiment,
            assets=assets,
            gt_coco_path=gt_coco_path,
            pred_coco_path=pred_coco_path,
            inference_type=self.inference_type,
        )

        # Sanitize IDs and match predictions
        gt_path_fixed = fix_coco_ids(gt_coco_path)
        pred_path_fixed = fix_coco_ids(pred_coco_path)
        matched_pred_path = pred_path_fixed.replace(".json", "_matched.json")
        match_image_ids(gt_path_fixed, pred_path_fixed, matched_pred_path)

        # Load COCO objects
        coco_gt = COCO(gt_path_fixed)
        coco_pred = COCO(matched_pred_path)

        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds())

        y_true = []
        y_pred = []

        image_to_gt = {}
        for ann in gt_anns:
            image_to_gt[ann["image_id"]] = ann["category_id"]

        for ann in pred_anns:
            img_id = ann["image_id"]
            if img_id in image_to_gt:
                y_true.append(image_to_gt[img_id])
                y_pred.append(ann["category_id"])

        if not y_true:
            logger.warning("No matching ground truth and predictions found.")
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Load class names
        id_to_name = {
            cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
        }

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        logger.info(
            f"[Classification] Accuracy={accuracy:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}"
        )

        # Per-class metrics
        class_report = classification_report(
            y_true,
            y_pred,
            target_names=[id_to_name[i] for i in sorted(id_to_name.keys())],
            output_dict=True,
            zero_division=0,
        )

        rows = []
        row_labels = []

        for class_name, metrics in class_report.items():
            row_labels.append(class_name)
            rows.append(
                [
                    round(metrics["precision"], 3),
                    round(metrics["recall"], 3),
                    round(metrics["f1-score"], 3),
                ]
            )

        self.experiment_logger.log_table(
            name="test/metrics",
            data={
                "data": rows,
                "rows": row_labels,
                "columns": ["Precision", "Recall", "F1-score"],
            },
            phase="test",
        )

        # Global summary as individual scalar values
        global_metrics = {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
        }

        for metric_name, metric_value in global_metrics.items():
            self.experiment_logger.log_value(
                name=metric_name, value=metric_value, phase="test"
            )

        cm = confusion_matrix(y_true, y_pred, labels=sorted(id_to_name.keys()))
        label_map = {i: id_to_name[i] for i in sorted(id_to_name.keys())}

        self.experiment_logger.log_confusion_matrix(
            name="confusion-matrix", labelmap=label_map, matrix=cm, phase="test"
        )
