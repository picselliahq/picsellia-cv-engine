import os
from typing import Any, TypeVar

from picsellia import Experiment, Label, ModelVersion

from picsellia_cv_engine.models.model import ModelDownloader


class Model:
    """
    Manages the context of a specific model version, including paths, weights, configuration, and labels.

    This class handles the organization of model-related files such as pretrained weights, trained weights,
    configuration files, and exported weights. It provides methods for downloading these files, managing
    the model's runtime instance, and storing model artifacts in an experiment.
    """

    def __init__(
        self,
        name: str,
        model_version: ModelVersion,
        pretrained_weights_name: str | None = None,
        trained_weights_name: str | None = None,
        config_name: str | None = None,
        exported_weights_name: str | None = None,
        labelmap: dict[str, Label] | None = None,
    ):
        """
        Initializes the Model, which manages the paths, version, and related information for a specific model.

        Args:
            name (str): The name of the model.
            model_version (ModelVersion): The version of the model, which contains the pretrained model and configuration.
            pretrained_weights_name (Optional[str], optional): The name of the pretrained weights file attached to the model version in Picsellia. Defaults to None.
            trained_weights_name (Optional[str], optional): The name of the trained weights file attached to the model version in Picsellia. Defaults to None.
            config_name (Optional[str], optional): The name of the configuration file attached to the model version in Picsellia. Defaults to None.
            exported_weights_name (Optional[str], optional): The name of the exported weights file attached to the model version in Picsellia. Defaults to None.
            labelmap (Optional[Dict[str, Label]], optional): A dictionary mapping category names to labels. Defaults to None.
        """
        self.name = name
        """The name of the model."""

        self.model_version = model_version
        """The version of the model from Picsellia."""

        self.pretrained_weights_name = pretrained_weights_name
        """The name of the pretrained weights file attached to the model version in Picsellia."""

        self.trained_weights_name = trained_weights_name
        """The name of the trained weights file attached to the model version in Picsellia."""

        self.config_name = config_name
        """The name of the configuration file attached to the model version in Picsellia."""

        self.exported_weights_name = exported_weights_name
        """The name of the exported weights file attached to the model version in Picsellia."""

        self.labelmap = labelmap or {}
        """A dictionary mapping category names to labels."""

        self.weights_dir: str | None = None
        """The directory where model weights are stored."""

        self.results_dir: str | None = None
        """The directory where model results are stored."""

        self.pretrained_weights_dir: str | None = None
        """The directory where pretrained weights are stored."""

        self.trained_weights_dir: str | None = None
        """The directory where trained weights are stored."""

        self.config_dir: str | None = None
        """The directory where model configuration files are stored."""

        self.exported_weights_dir: str | None = None
        """The directory where exported weights are stored."""

        self.pretrained_weights_path: str | None = None
        """The path to the pretrained weights file."""

        self.trained_weights_path: str | None = None
        """The path to the trained weights file."""

        self.config_path: str | None = None
        """The path to the model configuration file."""

        self.exported_weights_path: str | None = None
        """The path to the exported weights file."""

        self._loaded_model: Any | None = None
        """The loaded model instance."""

    @property
    def loaded_model(self) -> Any:
        """
        Returns the loaded model instance. Raises an error if the model is not yet loaded.

        Returns:
            Any: The loaded model instance.

        Raises:
            ValueError: If the model is not loaded, an error is raised.
        """
        if self._loaded_model is None:
            raise ValueError(
                "Model is not loaded. Please load the model before accessing it."
            )
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """
        Sets the provided model instance as the loaded model.

        Args:
            model (Any): The model instance to set as loaded.
        """
        self._loaded_model = model

    def download_weights(self, destination_dir: str) -> None:
        """
        Downloads the model's weights and configuration files to the specified destination path.

        This method organizes the model weights into separate directories for pretrained weights, trained weights,
        and configuration files. It then downloads the corresponding files from the model version.

        Args:
            destination_dir (str): The destination path where the model weights and related files will be downloaded.
        """
        downloader = ModelDownloader()

        # Set destination directories
        self.weights_dir = os.path.join(destination_dir, "weights")
        self.results_dir = os.path.join(destination_dir, "results")
        self.pretrained_weights_dir = os.path.join(
            self.weights_dir, "pretrained_weights"
        )
        self.trained_weights_dir = os.path.join(self.weights_dir, "trained_weights")
        self.config_dir = os.path.join(self.weights_dir, "config")
        self.exported_weights_dir = os.path.join(self.weights_dir, "exported_weights")

        # Create directories if they don't exist
        for directory in [
            self.weights_dir,
            self.results_dir,
            self.pretrained_weights_dir,
            self.trained_weights_dir,
            self.config_dir,
            self.exported_weights_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Download and process model files
        for model_file in self.model_version.list_files():
            if model_file.name == self.pretrained_weights_name:
                self.pretrained_weights_path = downloader.download_and_process(
                    model_file, self.pretrained_weights_dir
                )
            elif model_file.name == self.trained_weights_name:
                self.trained_weights_path = downloader.download_and_process(
                    model_file, self.trained_weights_dir
                )
            elif model_file.name == self.config_name:
                self.config_path = downloader.download_and_process(
                    model_file, self.config_dir
                )
            elif model_file.name == self.exported_weights_name:
                self.exported_weights_path = downloader.download_and_process(
                    model_file, self.exported_weights_dir
                )
            else:
                downloader.download_and_process(model_file, self.weights_dir)

    def save_artifact_to_experiment(
        self, experiment: Experiment, artifact_name: str, artifact_path: str
    ) -> None:
        """
        Saves the specified artifact to the provided experiment.

        This method stores the artifact (e.g., model weights or configuration) to the experiment.

        Args:
            experiment (Experiment): The experiment to which the artifact will be saved.
            artifact_name (str): The name to assign to the artifact in the experiment.
            artifact_path (str): The path to the artifact file.

        Raises:
            ValueError: If the artifact path does not exist.
        """
        if not os.path.exists(artifact_path):
            raise ValueError(f"Artifact path {artifact_path} does not exist.")
        experiment.store(
            name=artifact_name,
            path=artifact_path,
        )


TModel = TypeVar("TModel", bound=Model)
