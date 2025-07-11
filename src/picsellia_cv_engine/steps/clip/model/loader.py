import logging

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.core.services.model.utils import build_model_impl
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel

logger = logging.getLogger(__name__)


@step
def load_clip_model(
    pretrained_weights_name: str,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    exported_weights_name: str | None = None,
    repo_id: str = "openai/clip-vit-large-patch14-336",
) -> CLIPModel:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model = build_model_impl(
        context=context,
        model_cls=CLIPModel,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        exported_weights_name=exported_weights_name,
    )

    if not model.pretrained_weights_path:
        raise FileNotFoundError("No pretrained weights path found in model.")

    loaded_model, loaded_processor = model.load_weights(
        weights_path=model.pretrained_weights_path, repo_id=repo_id
    )
    model.set_loaded_model(loaded_model)
    model.set_loaded_processor(loaded_processor)

    return model
