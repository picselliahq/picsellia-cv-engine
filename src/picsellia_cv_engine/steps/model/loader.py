import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.models import Model
from picsellia_cv_engine.models.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)


@step
def load_model(
    pretrained_weights_name: str | None = None,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    exported_weights_name: str | None = None,
) -> Model:
    """
    Loads the model for training, including weight initialization.

    This function retrieves the model associated with the active training experiment and initializes a
    `Model` object. It also downloads the necessary model weights to a local directory.

    Args:
        pretrained_weights_name (str, optional): Name of the pretrained weights to use. Defaults to None.
        trained_weights_name (str, optional): Name of the trained weights to load. Defaults to None.
        config_name (str, optional): Name of the model configuration file. Defaults to None.
        exported_weights_name (str, optional): Name of the exported weights for inference. Defaults to None.

    Returns:
        Model: The initialized model with downloaded weights.

    Raises:
        ResourceNotFoundError: If the model version is not found in the experiment.
        IOError: If the weight files cannot be downloaded.
    """
    context = Pipeline.get_active_context()

    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        model_version = context.experiment.get_base_model_version()
        model = Model(
            name=model_version.name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
        )
        model.download_weights(
            destination_dir=os.path.join(os.getcwd(), context.experiment.name, "model")
        )
        return model
    elif isinstance(context, PicselliaProcessingContext | LocalProcessingContext):
        if context.model_version_id:
            model_version = context.model_version
            model = Model(
                name=model_version.name,
                model_version=model_version,
                pretrained_weights_name=pretrained_weights_name,
                trained_weights_name=trained_weights_name,
                config_name=config_name,
                exported_weights_name=exported_weights_name,
            )
            model.download_weights(
                destination_dir=os.path.join(os.getcwd(), context.job_id, "model")
            )
            return model
