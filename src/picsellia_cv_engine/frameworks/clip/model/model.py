from typing import Any

import torch
from picsellia import Label, ModelVersion
from transformers import CLIPModel as TransformerCLIPModel
from transformers import CLIPProcessor as TransformerCLIPProcessor

from picsellia_cv_engine.core.models import Model


class CLIPModel(Model):
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
        super().__init__(
            name=name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
            labelmap=labelmap,
        )

        self._loaded_processor: Any | None = None
        """The loaded processor instance."""

    @property
    def loaded_processor(self) -> Any:
        """
        Return the loaded processor instance.

        Raises:
            ValueError: If the processor is not yet loaded.
        """
        if self._loaded_processor is None:
            raise ValueError(
                "Processor is not loaded. Please load the processor before accessing it."
            )
        return self._loaded_processor

    def set_loaded_processor(self, processor: Any) -> None:
        """Set the runtime-loaded processor instance."""
        self._loaded_processor = processor

    def load_weights(
        self, weights_path: str, repo_id: str = "openai/clip-vit-large-patch14-336"
    ) -> tuple[TransformerCLIPModel, TransformerCLIPProcessor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerCLIPModel.from_pretrained(weights_path).to(device).eval()
        processor = TransformerCLIPProcessor.from_pretrained(repo_id)
        return model, processor
