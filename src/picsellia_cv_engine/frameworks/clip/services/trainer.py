import os

import torch

from picsellia_cv_engine.core import DatasetCollection
from picsellia_cv_engine.core.contexts import (
    LocalTrainingContext,
    PicselliaTrainingContext,
)

from .clip_utils import (
    export_dataset_to_clip_json,
    prepare_caption_model,
    run_clip_training,
    save_best_checkpoint,
)


class ClipModelTrainer:
    """
    Trainer class for CLIP fine-tuning using BLIP-generated captions.

    This class prepares the dataset with captions, then launches the CLIP training script and logs the results.
    """

    def __init__(
        self,
        context: PicselliaTrainingContext | LocalTrainingContext,
        model_dir: str,
        run_script_path: str,
    ):
        """
        Args:
            context: Training context with experiment, hyperparameters, etc.
            model_dir: Path where model outputs/checkpoints will be saved.
            run_script_path: Path to the `run_clip.py` script for training.
        """
        self.context = context
        self.model_dir = model_dir
        self.run_script_path = run_script_path

    def train_model(self, dataset_collection: DatasetCollection) -> None:
        """
        Executes full training cycle:
        - Generate captions with BLIP.
        - Export to CLIP JSON format.
        - Run training script.
        - Save best checkpoint.

        Args:
            dataset_collection: DatasetCollection with train/val/test sets.
        """
        working_dir = self.context.working_dir
        os.makedirs(json_dir := os.path.join(working_dir, "json"), exist_ok=True)

        json_files = {
            split: os.path.join(json_dir, f"{split}.json")
            for split in ["train", "val", "test"]
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model, processor = prepare_caption_model(device)

        for split, json_path in json_files.items():
            export_dataset_to_clip_json(
                model=blip_model,
                processor=processor,
                dataset=dataset_collection[split],
                output_path=json_path,
                device=device,
                prompt=self.context.hyperparameters.caption_prompt,
            )

        del blip_model
        torch.cuda.empty_cache()

        os.makedirs(self.model_dir, exist_ok=True)

        run_clip_training(
            run_script_path=self.run_script_path,
            output_dir=self.model_dir,
            train_json=json_files["train"],
            val_json=json_files["val"],
            test_json=json_files["test"],
            batch_size=self.context.hyperparameters.batch_size,
            epochs=self.context.hyperparameters.epochs,
            context=self.context,
        )

        save_best_checkpoint(output_dir=self.model_dir, context=self.context)
