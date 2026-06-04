from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from picsellia.types.enums import ProcessingType
from toml import load as load_toml

from picsellia_cv_engine.core.parameters.augmentation_parameters import (
    TAugmentationParameters,
)
from picsellia_cv_engine.core.parameters.base_parameters import TParameters
from picsellia_cv_engine.core.parameters.export_parameters import TExportParameters
from picsellia_cv_engine.core.parameters.hyper_parameters import THyperParameters
from picsellia_cv_engine.core.services.context.config import (
    AutoAnnotationConfig,
    BaseConfig,
    DataAugmentationConfig,
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
from picsellia_cv_engine.core.services.context.processing_groups import (
    DATASET_VERSION_OUTPUT_TYPES,
    PRE_ANNOTATION_LIKE_TYPES,
    is_datalake_processing,
    is_dataset_processing,
    is_model_processing,
)

Mode = Literal["local", "picsellia"]

ProcessingConfig: TypeAlias = (
    PreAnnotationConfig
    | AutoAnnotationConfig
    | DatasetVersionCreationConfig
    | DataAugmentationConfig
    | DataAutoTaggingConfig
    | ModelProcessConfig
)


def _load_and_validate_processing_config(
    config_file: str | Path, processing_type: ProcessingType
) -> ProcessingConfig:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = load_toml(path)

    if processing_type == ProcessingType.PRE_ANNOTATION:
        return PreAnnotationConfig(**raw)
    if processing_type == ProcessingType.AUTO_ANNOTATION:
        return AutoAnnotationConfig(**raw)
    if processing_type == ProcessingType.DATASET_VERSION_CREATION:
        return DatasetVersionCreationConfig(**raw)
    if processing_type == ProcessingType.DATA_AUGMENTATION:
        return DataAugmentationConfig(**raw)
    if is_datalake_processing(processing_type):
        return DataAutoTaggingConfig(**raw)
    if is_model_processing(processing_type):
        return ModelProcessConfig(**raw)

    raise RuntimeError(f"Unsupported processing type: {processing_type}")


def _load_and_validate_training_config(config_file: str | Path) -> TrainingConfig:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = load_toml(path)

    return TrainingConfig(**raw)


def _resolve_target_id_from_dataset_input(config: BaseConfig) -> str:
    if config.target_id:
        return config.target_id
    input_section = getattr(config, "input", None)
    if input_section is not None:
        dataset_version = getattr(input_section, "dataset_version", None)
        if dataset_version is not None and dataset_version.id:
            return dataset_version.id
    raise ValueError(
        "Missing target_id: set top-level target_id or input.dataset_version.id"
    )


def _resolve_target_id_from_datalake_input(config: DataAutoTaggingConfig) -> str:
    if config.target_id:
        return config.target_id
    if config.input and config.input.datalake and config.input.datalake.id:
        return config.input.datalake.id
    raise ValueError("Missing target_id: set top-level target_id or input.datalake.id")


def _resolve_target_id_from_model_input(config: ModelProcessConfig) -> str:
    if config.target_id:
        return config.target_id
    if config.input and config.input.model_version and config.input.model_version.id:
        return config.input.model_version.id
    raise ValueError(
        "Missing target_id: set top-level target_id or input.model_version.id"
    )


def _build_pre_annotation_like_local_context(
    config: PreAnnotationConfig | AutoAnnotationConfig,
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
):
    inputs: dict[str, Any] = dict(config.inputs) if config.inputs else {}
    target_id = _resolve_target_id_from_dataset_input(config)
    inputs["input_dataset_version_id"] = target_id

    if config.input and config.input.model_version and config.input.model_version.id:
        inputs["model_version_id"] = config.input.model_version.id
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


def _build_dataset_version_output_local_context(
    config: DatasetVersionCreationConfig | DataAugmentationConfig,
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
):
    inputs: dict[str, Any] = dict(config.inputs) if config.inputs else {}
    target_id = _resolve_target_id_from_dataset_input(config)
    inputs["input_dataset_version_id"] = target_id

    if (
        config.output
        and config.output.dataset_version
        and config.output.dataset_version.id
    ):
        inputs["output_dataset_version_id"] = config.output.dataset_version.id
    if (
        config.output
        and config.output.dataset_version
        and config.output.dataset_version.name
    ):
        inputs["target_version_name"] = config.output.dataset_version.name
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


def _build_datalake_local_context(
    config: DataAutoTaggingConfig,
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
):
    inputs: dict[str, Any] = dict(config.inputs) if config.inputs else {}
    target_id = _resolve_target_id_from_datalake_input(config)
    inputs["input_datalake_id"] = target_id

    if config.output and config.output.datalake and config.output.datalake.id:
        inputs["output_datalake_id"] = config.output.datalake.id
    if config.input and config.input.model_version and config.input.model_version.id:
        inputs["model_version_id"] = config.input.model_version.id
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


def _build_model_local_context(
    config: ModelProcessConfig,
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
):
    inputs: dict[str, Any] = dict(config.inputs) if config.inputs else {}
    target_id = _resolve_target_id_from_model_input(config)
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


def _create_local_processing_context_from_config(
    config: ProcessingConfig,
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
):
    if processing_type in PRE_ANNOTATION_LIKE_TYPES:
        return _build_pre_annotation_like_local_context(
            config=cast(PreAnnotationConfig | AutoAnnotationConfig, config),
            processing_type=processing_type,
            processing_parameters_cls=processing_parameters_cls,
        )
    if processing_type in DATASET_VERSION_OUTPUT_TYPES:
        return _build_dataset_version_output_local_context(
            config=cast(DatasetVersionCreationConfig | DataAugmentationConfig, config),
            processing_type=processing_type,
            processing_parameters_cls=processing_parameters_cls,
        )
    if is_datalake_processing(processing_type):
        return _build_datalake_local_context(
            config=cast(DataAutoTaggingConfig, config),
            processing_type=processing_type,
            processing_parameters_cls=processing_parameters_cls,
        )
    if is_model_processing(processing_type):
        return _build_model_local_context(
            config=cast(ModelProcessConfig, config),
            processing_type=processing_type,
            processing_parameters_cls=processing_parameters_cls,
        )
    raise RuntimeError(
        f"Unsupported processing type for local context: {processing_type}"
    )


def create_processing_context_from_config(
    processing_type: ProcessingType,
    processing_parameters_cls: type[TParameters],
    mode: Mode = "picsellia",
    config_file_path: str | Path | None = None,
):
    if mode == "picsellia":
        if is_dataset_processing(processing_type):
            return create_picsellia_dataset_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        if is_datalake_processing(processing_type):
            return create_picsellia_datalake_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        if is_model_processing(processing_type):
            return create_picsellia_model_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        raise RuntimeError(f"Unsupported processing type: {processing_type}")

    if config_file_path is None:
        raise ValueError("Config file path must be provided for local mode")

    config = _load_and_validate_processing_config(
        config_file=config_file_path, processing_type=processing_type
    )
    return _create_local_processing_context_from_config(
        config=config,
        processing_type=processing_type,
        processing_parameters_cls=processing_parameters_cls,
    )


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
