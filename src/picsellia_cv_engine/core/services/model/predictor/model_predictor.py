from abc import ABC
from typing import Generic, TypeVar

from picsellia_cv_engine.core import Model
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaRectangle,
)

TModel = TypeVar("TModel", bound=Model)


class ModelPredictor(ABC, Generic[TModel]):
    def __init__(self, model: TModel):
        """
        Initializes the base class for performing inference using a models.

        Args:
            model (TModel): The context containing the loaded models and configurations.
        """
        self.model: TModel = model

        if not hasattr(self.model, "loaded_model"):
            raise ValueError("The models does not have a loaded models attribute.")

    def get_picsellia_label(
        self, category_name: str, dataset: TBaseDataset
    ) -> PicselliaLabel:
        """
        Retrieves or creates a label for a given category name within the dataset.

        Args:
            category_name (str): The name of the category to retrieve the label for.
            dataset (TBaseDataset): The dataset containing the label information.

        Returns:
            PicselliaLabel: The corresponding Picsellia label for the given category.
        """
        return PicselliaLabel(
            dataset.dataset_version.get_or_create_label(category_name)
        )

    def get_picsellia_confidence(self, confidence: float) -> PicselliaConfidence:
        """
        Converts a confidence score into a PicselliaConfidence object.

        Args:
            confidence (float): The confidence score for the prediction.

        Returns:
            PicselliaConfidence: The confidence score wrapped in a PicselliaConfidence object.
        """
        return PicselliaConfidence(confidence)

    def get_picsellia_rectangle(
        self, x: int, y: int, w: int, h: int
    ) -> PicselliaRectangle:
        """
        Creates a PicselliaRectangle object representing a bounding box.

        Args:
            x (int): The x-coordinate of the top-left corner of the rectangle.
            y (int): The y-coordinate of the top-left corner of the rectangle.
            w (int): The width of the rectangle.
            h (int): The height of the rectangle.

        Returns:
            PicselliaRectangle: The rectangle object with the specified dimensions.
        """
        return PicselliaRectangle(x=x, y=y, w=w, h=h)
