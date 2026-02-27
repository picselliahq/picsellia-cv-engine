import os
from collections.abc import Callable
from typing import Any, TypeVar

from picsellia import Artifact, Experiment, Label, ModelFile, ModelVersion

from .model_downloader import ModelDownloader


class Model:
    """
    Represents a model version and manages its associated files and runtime instance.
    """

    def __init__(
        self,
        name: str,
        model_version: ModelVersion | None = None,
        experiment: Experiment | None = None,
        pretrained_weights_name: str | None = None,
        trained_weights_name: str | None = None,
        config_name: str | None = None,
        exported_weights_name: str | None = None,
        labelmap: dict[str, Label] | None = None,
    ):
        """
        Initialize the model with its version and associated files.
        """
        self.name = name
        """The name of the model."""

        self.model_version = model_version
        """The version of the model from Picsellia."""

        self.experiment = experiment
        """The Picsellia experiment associated with the model, used for logging artifacts and metrics."""

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
        Return the loaded model instance.

        Raises:
            ValueError: If the model is not yet loaded.
        """
        if self._loaded_model is None:
            raise ValueError(
                "Model is not loaded. Please load the model before accessing it."
            )
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """Set the runtime-loaded model instance."""
        self._loaded_model = model

    def download_model_weights(self, destination_dir: str) -> None:
        """
        Download all configured model files (weights, config, exports) to destination.

        Args:
            destination_dir (str): Root directory for downloaded files.
        """
        if self.model_version is None:
            raise ValueError(
                f"No model version available for model '{self.name}', cannot download files."
            )
        self._prepare_directories(destination_dir)
        self._do_download_files(dl_method=self.model_version.list_files)

    def download_experiment_weights(self, destination_dir: str) -> None:
        """
        Download all configured artifact files (weights, config, exports) to destination.

        Args:
            destination_dir (str): Root directory for downloaded files.
        """
        if self.experiment is None:
            raise ValueError(
                f"No experiment available '{self.name}', cannot download files."
            )
        self._prepare_directories(destination_dir)
        self._do_download_files(dl_method=self.experiment.list_artifacts)

    def _prepare_directories(self, destination_dir: str) -> None:
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

    def _do_download_files(  # noqa: C901
        self, dl_method: Callable[[], list[ModelFile] | list[Artifact]]
    ):
        downloader = ModelDownloader()
        # Download and process files
        for file in dl_method():
            if file.name == self.pretrained_weights_name:
                if not self.pretrained_weights_dir:
                    raise ValueError(
                        "Pretrained weights directory is not set. Cannot download pretrained weights."
                    )
                self.pretrained_weights_path = downloader.download_and_process(
                    file, self.pretrained_weights_dir
                )
            elif file.name == self.trained_weights_name:
                if not self.trained_weights_dir:
                    raise ValueError(
                        "Trained weights directory is not set. Cannot download trained weights."
                    )
                self.trained_weights_path = downloader.download_and_process(
                    file, self.trained_weights_dir
                )
            elif file.name == self.config_name:
                if not self.config_dir:
                    raise ValueError(
                        "Config directory is not set. Cannot download config file."
                    )
                self.config_path = downloader.download_and_process(
                    file, self.config_dir
                )
            elif file.name == self.exported_weights_name:
                if not self.exported_weights_dir:
                    raise ValueError(
                        "Exported weights directory is not set. Cannot download exported weights."
                    )
                self.exported_weights_path = downloader.download_and_process(
                    file, self.exported_weights_dir
                )
            else:
                if not self.weights_dir:
                    raise ValueError(
                        "Weights directory is not set. Cannot download model file."
                    )
                downloader.download_and_process(file, self.weights_dir)

    def save_artifact_to_experiment(
        self, artifact_name: str, artifact_path: str
    ) -> None:
        """
        Store an artifact file in the given experiment.

        Args:
            artifact_name (str): Name under which to save the artifact.
            artifact_path (str): Path to the file.

        Raises:
            ValueError: If the artifact path doesn't exist.
        """
        if not self.experiment:
            raise ValueError(
                f"No experiment associated with model '{self.name}', cannot save artifact."
            )
        if not os.path.exists(artifact_path):
            raise ValueError(f"Artifact path {artifact_path} does not exist.")
        self.experiment.store(
            name=artifact_name,
            path=artifact_path,
        )


TModel = TypeVar("TModel", bound=Model)
