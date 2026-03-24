from pathlib import Path
from typing import Literal

from picsellia.types.enums import ProcessingType
from toml import load as load_toml

from picsellia_cv_engine.core.parameters.augmentation_parameters import (
    TAugmentationParameters,
)
from picsellia_cv_engine.core.parameters.base_parameters import TParameters
from picsellia_cv_engine.core.parameters.export_parameters import TExportParameters
from picsellia_cv_engine.core.parameters.hyper_parameters import THyperParameters
from picsellia_cv_engine.core.services.context.config import (
    DataAutoTaggingConfig,
    DatasetVersionCreationConfig,
    ModelProcessConfig,
    PreAnnotationConfig,
    TrainingConfig,
)
from picsellia_cv_engine.core.services.context.local_context import (
    create_local_datalake_processing_context,
    create_local_dataset_processing_context,
    create_local_model_processing_context,
    create_local_training_context,
)
from picsellia_cv_engine.core.services.context.picsellia_context import (
    create_picsellia_datalake_processing_context,
    create_picsellia_dataset_processing_context,
    create_picsellia_model_processing_context,
    create_picsellia_training_context,
)

Mode = Literal["local", "picsellia"]


def _load_and_validate_processing_config(
    config_file: str | Path, processing_type: ProcessingType
) -> (
    PreAnnotationConfig
    | DatasetVersionCreationConfig
    | DataAutoTaggingConfig
    | ModelProcessConfig
):
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = load_toml(path)

    if processing_type == ProcessingType.PRE_ANNOTATION:
        return PreAnnotationConfig(**raw)
    elif processing_type == ProcessingType.DATASET_VERSION_CREATION:
        return DatasetVersionCreationConfig(**raw)
    elif processing_type == ProcessingType.DATA_AUTO_TAGGING:
        return DataAutoTaggingConfig(**raw)
    elif (
        processing_type == ProcessingType.MODEL_CONVERSION
        or processing_type == ProcessingType.MODEL_COMPRESSION
    ):
        return ModelProcessConfig(**raw)
    else:
        raise RuntimeError(f"Unsupported processing type: {processing_type}")


def _load_and_validate_training_config(config_file: str | Path) -> TrainingConfig:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = load_toml(path)

    return TrainingConfig(**raw)


def create_processing_context_from_config(  # noqa: C901
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
    mode: Mode = "picsellia",
    config_file_path: str | Path | None = None,
):
    if mode == "picsellia":
        if (
            processing_type == ProcessingType.PRE_ANNOTATION
            or processing_type == ProcessingType.DATASET_VERSION_CREATION
        ):
            return create_picsellia_dataset_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        elif processing_type == ProcessingType.DATA_AUTO_TAGGING:
            return create_picsellia_datalake_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        elif (
            processing_type == ProcessingType.MODEL_CONVERSION
            or processing_type == ProcessingType.MODEL_COMPRESSION
        ):
            return create_picsellia_model_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        else:
            raise RuntimeError(f"Unsupported processing type: {processing_type}")

    else:
        if config_file_path is None:
            raise ValueError("Config file path must be provided for local mode")
        config = _load_and_validate_processing_config(
            config_file=config_file_path, processing_type=processing_type
        )
        if processing_type == ProcessingType.PRE_ANNOTATION:
            inputs = {}
            if config.target_id:
                target_id = config.target_id
            else:
                target_id = config.input.dataset_version.id
            inputs["input_dataset_version_id"] = target_id
            if config.input and config.input.model_version.id:
                inputs["model_version_id"] = config.input.model_version.id
            elif config.inputs and "model_version_id" in config.inputs.keys():
                inputs["model_version_id"] = config.inputs["model_version_id"]
            return create_local_dataset_processing_context(
                processing_parameters_cls=processing_parameters_cls,
                organization_name=config.auth.organization_name,
                host=config.auth.host,
                job_type=processing_type,
                target_id=target_id,
                inputs=inputs,
                processing_parameters=dict(config.parameters),
                working_dir=config.run.working_dir,
            )

        elif processing_type == ProcessingType.DATASET_VERSION_CREATION:
            inputs = {}
            print("##### Config for dataset version creation:", config)
            if config.target_id:
                target_id = config.target_id
            else:
                target_id = config.input.dataset_version.id
            inputs["input_dataset_version_id"] = target_id
            if config.output and config.output.dataset_version.id:
                inputs["output_dataset_version_id"] = config.output.dataset_version.id
            elif config.inputs and "output_dataset_version_id" in config.inputs.keys():
                inputs["output_dataset_version_id"] = config.inputs[
                    "output_dataset_version_id"
                ]
            if config.output and config.output.dataset_version.name:
                inputs["target_version_name"] = config.output.dataset_version.name
            elif config.inputs and "target_version_name" in config.inputs.keys():
                inputs["target_version_name"] = config.inputs["target_version_name"]
            print("##### Inputs for dataset version creation:", inputs)
            return create_local_dataset_processing_context(
                processing_parameters_cls=processing_parameters_cls,
                organization_name=config.auth.organization_name,
                host=config.auth.host,
                job_type=processing_type,
                processing_parameters=dict(config.parameters),
                working_dir=config.run.working_dir,
                inputs=inputs,
                target_id=target_id,
            )

        elif processing_type == ProcessingType.DATA_AUTO_TAGGING:
            inputs = {}
            if config.target_id:
                target_id = config.target_id
            else:
                target_id = config.input.datalake.id
            inputs["input_datalake_id"] = target_id
            if config.output and config.output.datalake.id:
                inputs["output_datalake_id"] = config.output.datalake.id
            elif config.inputs and "output_datalake_id" in config.inputs.keys():
                inputs["output_datalake_id"] = config.inputs["output_datalake_id"]
            if config.input and config.input.model_version.id:
                inputs["model_version_id"] = config.input.model_version.id
            elif config.inputs and "model_version_id" in config.inputs.keys():
                inputs["model_version_id"] = config.inputs["model_version_id"]
            return create_local_datalake_processing_context(
                processing_parameters_cls=processing_parameters_cls,
                organization_name=config.auth.organization_name,
                host=config.auth.host,
                job_type=processing_type,
                target_id=target_id,
                offset=config.run_parameters.offset,
                limit=config.run_parameters.limit,
                processing_parameters=dict(config.parameters),
                working_dir=config.run.working_dir,
                inputs=inputs,
            )
        elif (
            processing_type == ProcessingType.MODEL_CONVERSION
            or processing_type == ProcessingType.MODEL_COMPRESSION
        ):
            inputs = {}
            if config.target_id:
                target_id = config.target_id
            else:
                target_id = config.input.model_version.id
            inputs["input_model_version_id"] = target_id
            return create_local_model_processing_context(
                processing_parameters_cls=processing_parameters_cls,
                organization_name=config.auth.organization_name,
                host=config.auth.host,
                job_type=processing_type,
                target_id=target_id,
                processing_parameters=dict(config.parameters),
                working_dir=config.run.working_dir,
                inputs=inputs,
            )
        else:
            raise RuntimeError("Unsupported processing type for local context")


def create_training_context_from_config(
    hyperparameters_cls: type[THyperParameters],
    augmentation_parameters_cls: type[TAugmentationParameters],
    export_parameters_cls: type[TExportParameters],
    mode: Mode = "picsellia",
    config_file_path: str | Path | None = None,
):
    if mode == "picsellia":
        return create_picsellia_training_context(
            hyperparameters_cls=hyperparameters_cls,
            augmentation_parameters_cls=augmentation_parameters_cls,
            export_parameters_cls=export_parameters_cls,
        )
    else:
        if config_file_path is None:
            raise ValueError("Config file path must be provided for local mode")
        config = _load_and_validate_training_config(config_file=config_file_path)
        return create_local_training_context(
            hyperparameters_cls=hyperparameters_cls,
            augmentation_parameters_cls=augmentation_parameters_cls,
            export_parameters_cls=export_parameters_cls,
            organization_name=config.auth.organization_name,
            host=config.auth.host,
            experiment_id=config.output.experiment.id,
            hyperparameters=dict(config.hyperparameters),
            augmentation_parameters=dict(config.augmentations_parameters),
            export_parameters=dict(config.export_parameters),
            working_dir=config.run.working_dir,
        )
