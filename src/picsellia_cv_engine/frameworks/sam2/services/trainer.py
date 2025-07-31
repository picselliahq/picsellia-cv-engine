import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

from picsellia.types.enums import LogType
from PIL import Image, ImageDraw

from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model


class Sam2Trainer:
    def __init__(
        self,
        model: Model,
        dataset_collection: DatasetCollection[CocoDataset],
        context,
        sam2_root: str,
    ):
        self.model = model
        self.dataset_collection = dataset_collection
        self.context = context
        self.sam2_root = sam2_root

        self.img_root = os.path.join(sam2_root, "data", "JPEGImages")
        self.ann_root = os.path.join(sam2_root, "data", "Annotations")
        prepare_directories(self.img_root, self.ann_root)

    def prepare_data(self):
        source_images = self.dataset_collection["train"].images_dir
        source_annotations = self.dataset_collection["train"].annotations_dir
        coco_file = next(
            f for f in os.listdir(source_annotations) if f.endswith(".json")
        )
        coco_path = os.path.join(source_annotations, coco_file)
        shutil.copy(
            coco_path, os.path.join(self.context.working_dir, "coco_annotations.json")
        )

        shutil.copy(
            self.model.pretrained_weights_path,
            os.path.join(self.sam2_root, "checkpoints"),
        )
        pretrained_weights_name = os.path.basename(self.model.pretrained_weights_path)
        self.model.pretrained_weights_path = os.path.join(
            self.sam2_root, "checkpoints", pretrained_weights_name
        )

        coco = load_coco_annotations(
            os.path.join(self.context.working_dir, "coco_annotations.json")
        )
        convert_coco_to_png_masks(coco, source_images, self.img_root, self.ann_root)
        normalize_filenames([self.img_root, self.ann_root])

        return pretrained_weights_name

    def launch_training(self, pretrained_weights_name):
        experiment_log_dir = os.path.join(self.model.results_dir, "sam2_logs")
        os.makedirs(experiment_log_dir, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                self.sam2_root,
                os.path.join(self.sam2_root, "training"),
            ]
        )

        log_file = os.path.join(experiment_log_dir, "train_stdout.log")

        overrides = [
            f"scratch.train_batch_size={self.context.hyperparameters.batch_size}",
            f"scratch.resolution={self.context.hyperparameters.image_size}",
            f"scratch.base_lr={self.context.hyperparameters.learning_rate}",
            f"scratch.num_epochs={self.context.hyperparameters.epochs}",
            f"trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path=checkpoints/{pretrained_weights_name}",
        ]

        self.model.results_dir = os.path.join(
            self.sam2_root, "sam2_logs", "configs", "train.yaml"
        )

        command = [
            sys.executable,
            "-m",
            "training.train",
            "-c",
            "configs/train.yaml",
            "--use-cluster",
            "0",
            "--num-gpus",
            "1",
            *overrides,
        ]

        process = subprocess.Popen(
            command,
            cwd=self.sam2_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        parse_and_log_sam2_output(process, self.context, log_file)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        return os.path.join(self.model.results_dir, "checkpoints", "checkpoint.pt")

    def save_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.model.save_artifact_to_experiment(
                experiment=self.context.experiment,
                artifact_name="model-latest",
                artifact_path=checkpoint_path,
            )
            self.model.trained_weights_path = checkpoint_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def prepare_directories(img_root: str, ann_root: str):
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(ann_root, exist_ok=True)


def load_coco_annotations(coco_path: str) -> dict[str, Any]:
    with open(coco_path) as f:
        return json.load(f)


def generate_mask(width, height, annotations) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    object_idx = 1
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1 or "segmentation" not in ann:
            continue
        for seg in ann["segmentation"]:
            if isinstance(seg, list) and len(seg) >= 6:
                poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw.polygon(poly, fill=object_idx)
                object_idx += 1
    return mask


def convert_coco_to_png_masks(coco, source_images, img_root, ann_root):
    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image: dict[str, Any] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, annotations in annotations_by_image.items():
        img_info = images_by_id[img_id]
        width, height = img_info["width"], img_info["height"]
        original_file = img_info["file_name"]
        base_name = os.path.splitext(original_file)[0]

        video_img_dir = os.path.join(img_root, base_name)
        video_ann_dir = os.path.join(ann_root, base_name)
        os.makedirs(video_img_dir, exist_ok=True)
        os.makedirs(video_ann_dir, exist_ok=True)

        shutil.copy(
            os.path.join(source_images, original_file),
            os.path.join(video_img_dir, "00000.jpg"),
        )

        mask = generate_mask(width, height, annotations)
        mask.save(os.path.join(video_ann_dir, "00000.png"))


def normalize_filenames(root_dirs: list[str]):
    for root in root_dirs:
        for subdir, _, files in os.walk(root):
            for name in files:
                new_name = name.replace(".", "_", name.count(".") - 1)
                if not re.search(r"_\d+\.\w+$", new_name):
                    new_name = new_name.replace(".", "_1.")
                os.rename(os.path.join(subdir, name), os.path.join(subdir, new_name))


def parse_and_log_sam2_output(process, context, log_file_path):
    """
    Parse les sorties du training SAM2 et logge dynamiquement les métriques Picsellia
    avec des noms nettoyés.
    """

    meter_pattern = re.compile(r"Losses and meters:\s+({.*})")

    METRIC_NAME_MAPPING = {
        "Losses/train_all_loss": "train/total_loss",
        "Losses/train_all_loss_mask": "train/loss_mask",
        "Losses/train_all_loss_dice": "train/loss_dice",
        "Losses/train_all_loss_iou": "train/loss_iou",
        "Losses/train_all_loss_class": "train/loss_class",
        "Losses/train_all_core_loss": "train/loss_core",
        "Trainer/epoch": "train/epoch",
        "Trainer/steps_train": "train/step",
    }

    SKIPPED_METRICS = {"Trainer/where"}

    with open(log_file_path, "w") as log_file:
        for line in process.stdout:
            print(line, end="")  # stdout live
            log_file.write(line)

            match = meter_pattern.search(line)
            if match:
                try:
                    metrics_str = match.group(1)
                    metrics = json.loads(metrics_str.replace("'", '"'))  # JSON-safe

                    for name, value in metrics.items():
                        if name in SKIPPED_METRICS or not isinstance(
                            value, float | int
                        ):
                            continue

                        log_name = METRIC_NAME_MAPPING.get(
                            name, f"train/{name.replace('/', '_')}"
                        )
                        context.experiment.log(
                            name=log_name,
                            data=value,
                            type=LogType.LINE,
                        )
                except Exception as e:
                    print(f"⚠️ Erreur parsing métriques SAM2 : {e}")
