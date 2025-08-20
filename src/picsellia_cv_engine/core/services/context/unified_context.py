from typing import Literal, TypeVar

from picsellia.types.enums import ProcessingType

from picsellia_cv_engine.core.parameters import Parameters
from picsellia_cv_engine.core.services.context.config import (
    DataAutoTaggingConfig,
    DatasetVersionCreationConfig,
    PreAnnotationConfig,
    ProcessingConfig,
)
from picsellia_cv_engine.core.services.context.local_context import (
    create_local_datalake_processing_context,
    create_local_dataset_processing_context,
)
from picsellia_cv_engine.core.services.context.picsellia_context import (
    create_picsellia_datalake_processing_context,
    create_picsellia_dataset_processing_context,
)

Mode = Literal["local", "picsellia"]
TParameters = TypeVar("TParameters", bound=Parameters)


def create_processing_context_from_config(
    cfg: ProcessingConfig,
    processing_parameters_cls: type[TParameters],
    mode: Mode = "picsellia",
):
    if mode == "picsellia":
        if isinstance(cfg, PreAnnotationConfig) or isinstance(
            cfg, DatasetVersionCreationConfig
        ):
            return create_picsellia_dataset_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )
        elif isinstance(cfg, DataAutoTaggingConfig):
            return create_picsellia_datalake_processing_context(
                processing_parameters_cls=processing_parameters_cls,
            )

    if isinstance(cfg, PreAnnotationConfig):
        return create_local_dataset_processing_context(
            processing_parameters_cls=processing_parameters_cls,
            api_token=cfg.auth.api_token or "",
            organization_name=cfg.auth.organization_name,
            host=cfg.run.host,
            job_type=ProcessingType.PRE_ANNOTATION,
            input_dataset_version_id=cfg.io.input_dataset_version_id,
            output_dataset_version_name=None,
            model_version_id=cfg.model.model_version_id,
            processing_parameters=dict(cfg.parameters),
            working_dir=cfg.run.working_dir,
        )

    if isinstance(cfg, DatasetVersionCreationConfig):
        return create_local_dataset_processing_context(
            processing_parameters_cls=processing_parameters_cls,
            api_token=cfg.auth.api_token or "",
            organization_name=cfg.auth.organization_name,
            host=cfg.run.host,
            job_type=ProcessingType.DATASET_VERSION_CREATION,
            input_dataset_version_id=cfg.io.input_dataset_version_id,
            output_dataset_version_name=cfg.io.output_dataset_version_name,
            model_version_id=None,
            processing_parameters=dict(cfg.parameters),
            working_dir=cfg.run.working_dir,
        )

    if isinstance(cfg, DataAutoTaggingConfig):
        return create_local_datalake_processing_context(
            processing_parameters_cls=processing_parameters_cls,
            api_token=cfg.auth.api_token or "",
            organization_name=cfg.auth.organization_name,
            host=cfg.run.host,
            job_type=ProcessingType.DATA_AUTO_TAGGING,
            model_version_id=cfg.model.model_version_id,
            offset=cfg.run_parameters.offset,
            limit=cfg.run_parameters.limit,
            processing_parameters=dict(cfg.parameters),
            working_dir=cfg.run.working_dir,
        )

    raise RuntimeError("Unsupported processing config type")
