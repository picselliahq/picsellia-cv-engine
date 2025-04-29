from picsellia import Experiment

from picsellia_cv_engine.core.services.model.logging import (
    BaseLogger,
    Metric,
)
from picsellia_cv_engine.frameworks.ultralytics.services.model.logger.object_detection import (
    UltralyticsObjectDetectionMetricMapping,
)


class UltralyticsSegmentationMetricMapping(UltralyticsObjectDetectionMetricMapping):
    """ """

    def __init__(self):
        """ """
        super().__init__()
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="seg_loss", framework_name="train/seg_loss"),
        )

        self.add_metric(
            phase="val",
            metric=Metric(standard_name="seg_loss", framework_name="val/seg_loss"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="precision(M)", framework_name="metrics/precision(M)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="recall(M)", framework_name="metrics/recall(M)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(standard_name="mAP50(M)", framework_name="metrics/mAP50(M)"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50-95(M)", framework_name="metrics/mAP50-95(M)"
            ),
        )


class UltralyticsSegmentationLogger(BaseLogger):
    """ """

    def __init__(
        self,
        experiment: Experiment,
        metric_mapping: UltralyticsSegmentationMetricMapping,
    ):
        """ """
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)
