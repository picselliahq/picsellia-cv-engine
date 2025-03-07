import os
from typing import Any, Generic, TypeVar

from picsellia_cv_engine.models.model.model_context import ModelContext

TModelContext = TypeVar("TModelContext", bound=ModelContext)


class ModelCollection(Generic[TModelContext]):
    """
    A collection of model contexts for managing multiple models in a sequential manner.

    This class holds multiple model contexts and allows managing a single 'loaded' model
    at a time for the collection.

    Attributes:
        models (dict): A dictionary containing model contexts, where keys are model names.
        loaded_model (Optional[Any]): The currently loaded model for this collection.
    """

    def __init__(self, models: list[TModelContext]):
        """
        Initializes the collection with a list of model contexts.

        Args:
            models (List[TModelContext]): A list of model contexts.
        """
        self.models = {model.model_name: model for model in models}
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

    def __getitem__(self, key: str) -> TModelContext:
        """
        Retrieves a model context by its name.

        Args:
            key (str): The name of the model context.

        Returns:
            TModelContext: The model context corresponding to the given name.
        """
        return self.models[key]

    def __setitem__(self, key: str, value: TModelContext):
        """
        Sets or updates a model context in the collection.

        Args:
            key (str): The name of the model context to update or add.
            value (TModelContext): The model context object to associate with the given name.
        """
        self.models[key] = value

    def __iter__(self):
        """
        Iterates over all model contexts in the collection.

        Returns:
            Iterator: An iterator over the model contexts.
        """
        return iter(self.models.values())

    def download_weights(self, destination_dir: str) -> None:
        """
        Downloads weights for all models in the collection.
        """
        for model_context in self:
            model_context.download_weights(
                destination_dir=os.path.join(destination_dir, model_context.model_name)
            )


TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)
