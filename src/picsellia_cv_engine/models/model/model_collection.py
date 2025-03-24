import os
from typing import Any, Generic, TypeVar

from .model import Model

TModel = TypeVar("TModel", bound=Model)


class ModelCollection(Generic[TModel]):
    """
    A collection of models for managing multiple models in a sequential manner.

    This class holds multiple models and allows managing a single 'loaded' model
    at a time for the collection.

    Attributes:
        models (dict): A dictionary containing models, where keys are model names.
        loaded_model (Optional[Any]): The currently loaded model for this collection.
    """

    def __init__(self, models: list[TModel]):
        """
        Initializes the collection with a list of models.

        Args:
            models (List[TModel]): A list of models.
        """
        self.models = {model.name: model for model in models}
        self._loaded_model: Any | None = None

    @property
    def loaded_model(self) -> Any:
        """
        Returns the loaded model instance. Raises an error if no model is currently loaded.

        Returns:
            Any: The loaded model instance.

        Raises:
            ValueError: If no model is currently loaded.
        """
        if self._loaded_model is None:
            raise ValueError("No model is currently loaded in this collection.")
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """
        Sets the provided model instance as the loaded model for this collection.

        Args:
            model (Any): The model instance to set as loaded.
        """
        self._loaded_model = model

    def __getitem__(self, key: str) -> TModel:
        """
        Retrieves a model by its name.

        Args:
            key (str): The name of the model.

        Returns:
            TModel: The model corresponding to the given name.
        """
        return self.models[key]

    def __setitem__(self, key: str, value: TModel):
        """
        Sets or updates a model in the collection.

        Args:
            key (str): The name of the model to update or add.
            value (TModel): The model object to associate with the given name.
        """
        self.models[key] = value

    def __iter__(self):
        """
        Iterates over all models in the collection.

        Returns:
            Iterator: An iterator over the models.
        """
        return iter(self.models.values())

    def download_weights(self, destination_dir: str) -> None:
        """
        Downloads weights for all models in the collection.
        """
        for model in self:
            model.download_weights(
                destination_dir=os.path.join(destination_dir, model.name)
            )


TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)
