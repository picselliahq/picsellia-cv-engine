from dataclasses import dataclass

from picsellia import Asset, Label


@dataclass
class PicselliaLabel:
    """Label associated with a prediction."""

    value: Label


@dataclass
class PicselliaConfidence:
    """Confidence score for a prediction (typically between 0 and 1)."""

    value: float


@dataclass
class PicselliaRectangle:
    """Bounding box in [x, y, width, height] format."""

    value: list[int]

    def __init__(self, x: int, y: int, w: int, h: int):
        """Initialize rectangle coordinates."""
        self.value = [int(x), int(y), int(w), int(h)]


@dataclass
class PicselliaText:
    """Recognized text from OCR predictions."""

    value: str


@dataclass
class PicselliaPolygon:
    """Polygon represented by a list of points."""

    value: list[list[int]]

    def __init__(self, points: list[list[int]]):
        """Initialize polygon with a list of [x, y] points."""
        self.value = points


@dataclass
class PicselliaClassificationPrediction:
    """Prediction result for classification tasks."""

    asset: Asset
    label: PicselliaLabel
    confidence: PicselliaConfidence


@dataclass
class PicselliaRectanglePrediction:
    """Prediction result for object detection (rectangles)."""

    asset: Asset
    boxes: list[PicselliaRectangle]
    labels: list[PicselliaLabel]
    confidences: list[PicselliaConfidence]


@dataclass
class PicselliaOCRPrediction:
    """Prediction result for OCR tasks."""

    asset: Asset
    boxes: list[PicselliaRectangle]
    labels: list[PicselliaLabel]
    texts: list[PicselliaText]
    confidences: list[PicselliaConfidence]


@dataclass
class PicselliaPolygonPrediction:
    """Prediction result for segmentation tasks."""

    asset: Asset
    polygons: list[PicselliaPolygon]
    labels: list[PicselliaLabel]
    confidences: list[PicselliaConfidence]
