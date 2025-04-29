import logging

import torch
from ultralytics import YOLO

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.core.steps.model.builder import build_model_impl
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from picsellia_cv_engine.frameworks.ultralytics.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.frameworks.ultralytics.parameters.hyper_parameters import (
    UltralyticsHyperParameters,
)

logger = logging.getLogger(__name__)


@step
def load_ultralytics_model(
    pretrained_weights_name: str,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    exported_weights_name: str | None = None,
) -> UltralyticsModel:
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model = build_model_impl(
        context=context,
        model_cls=UltralyticsModel,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        exported_weights_name=exported_weights_name,
    )

    if not model.pretrained_weights_path:
        raise FileNotFoundError("No pretrained weights path found in model.")

    loaded_model = load_yolo_weights(
        weights_path_to_load=model.pretrained_weights_path,
        device=context.hyperparameters.device,
    )
    model.set_loaded_model(loaded_model)
    return model


def load_yolo_weights(weights_path_to_load: str, device: str) -> YOLO:
    """
    Loads a YOLO model from the given weights file and moves it to the specified device.

    This function loads a YOLO model using the provided weights path and transfers it
    to the specified device (e.g., 'cpu' or 'cuda'). It raises an error if the weights
    file is not found or cannot be loaded.

    Args:
        weights_path_to_load (str): The file path to the YOLO model weights.
        device (str): The device to which the model should be moved ('cpu' or 'cuda').

    Returns:
        YOLO: The loaded YOLO model ready for inference or training.

    Raises:
        RuntimeError: If the weights file cannot be loaded or the device is unavailable.
    """
    loaded_model = YOLO(weights_path_to_load)
    torch_device = torch.device(device)
    logger.info(f"Loading model on device: {torch_device}")
    loaded_model.to(device=device)
    return loaded_model
