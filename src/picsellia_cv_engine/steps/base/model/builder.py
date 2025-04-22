import os
from typing import TypeVar

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import Model
from picsellia_cv_engine.core.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)

TModel = TypeVar("TModel", bound=Model)


@step
def build_model(
    model_cls: type[TModel],
    pretrained_weights_name: str | None = None,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    exported_weights_name: str | None = None,
) -> TModel:
    """
    Loads the models for training, including weight initialization.

    This function retrieves the models associated with the active training experiment and initializes a
    `Model` object. It also downloads the necessary models weights to a local directory.

    Args:
        pretrained_weights_name (str, optional): Name of the pretrained weights to use. Defaults to None.
        trained_weights_name (str, optional): Name of the trained weights to load. Defaults to None.
        config_name (str, optional): Name of the models configuration file. Defaults to None.
        exported_weights_name (str, optional): Name of the exported weights for inference. Defaults to None.

    Returns:
        TModel: The initialized models with downloaded weights.

    Raises:
        ResourceNotFoundError: If the models version is not found in the experiment.
        IOError: If the weight files cannot be downloaded.
    """
    context = Pipeline.get_active_context()
    return build_model_impl(
        context=context,
        model_cls=model_cls,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        exported_weights_name=exported_weights_name,
    )


def build_model_impl(
    context,
    model_cls: type[TModel],
    pretrained_weights_name: str | None = None,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    exported_weights_name: str | None = None,
) -> TModel:
    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        model_version = context.experiment.get_base_model_version()
        destination_dir = os.path.join(os.getcwd(), context.experiment.name, "models")
    elif isinstance(context, PicselliaProcessingContext | LocalProcessingContext):
        if context.model_version_id:
            model_version = context.model_version
            destination_dir = os.path.join(os.getcwd(), context.job_id, "models")
        else:
            raise ValueError("No model_version_id provided in the processing context.")
    else:
        raise ValueError("The current context is not a training or processing context.")

    model = model_cls(
        name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        exported_weights_name=exported_weights_name,
    )
    model.download_weights(destination_dir=destination_dir)
    return model
