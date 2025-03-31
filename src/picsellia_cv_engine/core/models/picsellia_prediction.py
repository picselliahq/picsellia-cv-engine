from dataclasses import dataclass

from picsellia import Asset, Label


@dataclass
class PicselliaLabel:
    """
    Represents a label prediction in Picsellia.

    Attributes:
        value (Label): The label value associated with a prediction.
    """

    value: Label


@dataclass
class PicselliaConfidence:
    """
    Represents the confidence score of a prediction.

    Attributes:
        value (float): The confidence value associated with a prediction, typically between 0 and 1.
    """

    value: float


@dataclass
class PicselliaRectangle:
    """
    Represents a bounding box in the form of a rectangle.

    Attributes:
        value (list[int]): The list of coordinates [x, y, width, height] that define the rectangle.
    """

    value: list[int]

    def __init__(self, x: int, y: int, w: int, h: int):
        """
        Initializes the bounding box with the given coordinates.

        Args:
            x (int): The x-coordinate of the top-left corner.
            y (int): The y-coordinate of the top-left corner.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
        """
        self.value = [int(x), int(y), int(w), int(h)]


@dataclass
class PicselliaText:
    """
    Represents text information in OCR predictions.

    Attributes:
        value (str): The recognized text from the OCR process.
    """

    value: str


@dataclass
class PicselliaPolygon:
    """
    Represents a polygon with a series of points.

    Attributes:
        value (list[int]): A list of integer coordinates that define the polygon.
    """

    value: list[list[int]]

    def __init__(self, points: list[list[int]]):
        """
        Initializes the polygon with the given list of points.

        Args:
            points (list[list[int]]): A list of integer coordinates that define the polygon.
        """
        self.value = points


@dataclass
class PicselliaClassificationPrediction:
    """
    Represents a classification prediction for an asset.

    Attributes:
        asset (Asset): The asset associated with this prediction.
        label (PicselliaLabel): The predicted label.
        confidence (PicselliaConfidence): The confidence score of the prediction.
    """

    asset: Asset
    label: PicselliaLabel
    confidence: PicselliaConfidence


@dataclass
class PicselliaRectanglePrediction:
    """
    Represents a rectangle prediction for an asset, typically used in object detection.

    Attributes:
        asset (Asset): The asset associated with this prediction.
        boxes (list[PicselliaRectangle]): The bounding boxes predicted for the asset.
        labels (list[PicselliaLabel]): The labels corresponding to each bounding box.
        confidences (list[PicselliaConfidence]): The confidence scores for each predicted box.
    """

    asset: Asset
    boxes: list[PicselliaRectangle]
    labels: list[PicselliaLabel]
    confidences: list[PicselliaConfidence]


@dataclass
class PicselliaOCRPrediction:
    """
    Represents an OCR prediction for an asset, which includes bounding boxes, text, and labels.

    Attributes:
        asset (Asset): The asset associated with this OCR prediction.
        boxes (list[PicselliaRectangle]): The bounding boxes for text regions in the asset.
        labels (list[PicselliaLabel]): The labels corresponding to each text region.
        texts (list[PicselliaText]): The recognized text in each region.
        confidences (list[PicselliaConfidence]): The confidence scores for each prediction.
    """

    asset: Asset
    boxes: list[PicselliaRectangle]
    labels: list[PicselliaLabel]
    texts: list[PicselliaText]
    confidences: list[PicselliaConfidence]


@dataclass
class PicselliaPolygonPrediction:
    """
    Represents a polygon prediction for an asset, typically used in segmentation tasks.

    Attributes:
        asset (Asset): The asset associated with this prediction.
        polygons (list[PicselliaPolygon]): The predicted polygons.
        labels (list[PicselliaLabel]): The labels corresponding to each polygon.
        confidences (list[PicselliaConfidence]): The confidence scores for each predicted polygon.
    """

    asset: Asset
    polygons: list[PicselliaPolygon]
    labels: list[PicselliaLabel]
    confidences: list[PicselliaConfidence]
