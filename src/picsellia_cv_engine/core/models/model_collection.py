import os
from typing import Any, Generic, TypeVar

from .model import Model

TModel = TypeVar("TModel", bound=Model)


class ModelCollection(Generic[TModel]):
    """
    A collection of core for managing multiple core in a sequential manner.

    This class holds multiple core and allows managing a single 'loaded' models
    at a time for the collection.

    Attributes:
        models (dict): A dictionary containing core, where keys are models names.
        loaded_model (Optional[Any]): The currently loaded models for this collection.
    """

    def __init__(self, models: list[TModel]):
        """
        Initializes the collection with a list of core.

        Args:
            models (List[TModel]): A list of core.
        """
        self.models = {model.name: model for model in models}
        self._loaded_model: Any | None = None

    @property
    def loaded_model(self) -> Any:
        """
        Returns the loaded models instance. Raises an error if no models is currently loaded.

        Returns:
            Any: The loaded models instance.

        Raises:
            ValueError: If no models is currently loaded.
        """
        if self._loaded_model is None:
            raise ValueError("No models is currently loaded in this collection.")
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """
        Sets the provided models instance as the loaded models for this collection.

        Args:
            model (Any): The models instance to set as loaded.
        """
        self._loaded_model = model

    def __getitem__(self, key: str) -> TModel:
        """
        Retrieves a models by its name.

        Args:
            key (str): The name of the models.

        Returns:
            TModel: The models corresponding to the given name.
        """
        return self.models[key]

    def __setitem__(self, key: str, value: TModel):
        """
        Sets or updates a models in the collection.

        Args:
            key (str): The name of the models to update or add.
            value (TModel): The models object to associate with the given name.
        """
        self.models[key] = value

    def __iter__(self):
        """
        Iterates over all core in the collection.

        Returns:
            Iterator: An iterator over the core.
        """
        return iter(self.models.values())

    def download_weights(self, destination_dir: str) -> None:
        """
        Downloads weights for all core in the collection.
        """
        for model in self:
            model.download_weights(
                destination_dir=os.path.join(destination_dir, model.name)
            )


TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)
