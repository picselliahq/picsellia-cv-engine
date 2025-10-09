from typing import Any, Literal

from pydantic import BaseModel, Field


class Auth(BaseModel):
    organization_name: str
    env: str | None = None
    host: str | None = None


class Run(BaseModel):
    name: str | None = None
    working_dir: str | None = None
    mode: Literal["local", "picsellia"] | None = None


class Experiment(BaseModel):
    id: str | None = None
    name: str | None = None
    project_name: str | None = None
    url: str | None = None


class ModelVersion(BaseModel):
    id: str
    name: str | None = None
    origin_name: str | None = None
    url: str | None = None
    visibility: Literal["private", "public"]


class DatasetVersion(BaseModel):
    id: str | None = None
    name: str | None = None
    origin_name: str | None = None
    version_name: str | None = None
    url: str | None = None


class Datalake(BaseModel):
    id: str
    name: str | None = None
    url: str | None = None


# ── Jobs ──────────────────────────────────────────────────────────────────────


class JobTraining(BaseModel):
    type: Literal["TRAINING"]


class JobPreAnn(BaseModel):
    type: Literal["PRE_ANNOTATION"]


class JobDSVCreate(BaseModel):
    type: Literal["DATASET_VERSION_CREATION"]


class JobAutoTag(BaseModel):
    type: Literal["DATA_AUTO_TAGGING"]


class JobModelProcess(BaseModel):
    type: Literal["MODEL_CONVERSION", "MODEL_COMPRESSION"]


# ── Shared toggle ─────────────────────────────────────────────────────────────


class OverrideOutputsMixin(BaseModel):
    """Shared toggle to overwrite/replace existing outputs without prompting."""

    override_outputs: bool = Field(
        default=False,
        description=(
            "If true, existing target outputs (e.g., experiment bindings, dataset "
            "versions, target datalakes) will be overwritten or recreated without "
            "confirmation prompts where applicable."
        ),
    )


# ── DATASET_VERSION_CREATION ─────────────────────────────────────────────────


class InputDatasetVersionCreation(BaseModel):
    dataset_version: DatasetVersion


class OutputDatasetVersionCreation(BaseModel):
    dataset_version: DatasetVersion


class DatasetVersionCreationConfig(OverrideOutputsMixin, BaseModel):
    job: JobDSVCreate
    auth: Auth
    run: Run = Run()
    input: InputDatasetVersionCreation
    output: OutputDatasetVersionCreation
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── PRE_ANNOTATION ───────────────────────────────────────────────────────────


class InputPreAnnotation(BaseModel):
    dataset_version: DatasetVersion | None = None
    model_version: ModelVersion | None = None


class PreAnnotationConfig(OverrideOutputsMixin, BaseModel):
    job: JobPreAnn
    auth: Auth
    run: Run = Run()
    input: InputPreAnnotation
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── DATA_AUTO_TAGGING ────────────────────────────────────────────────────────


class AutoTagRunParams(BaseModel):
    offset: int = 0
    limit: int = 100


class InputDataAutoTagging(BaseModel):
    datalake: Datalake
    model_version: ModelVersion | None = None


class OutputDataAutoTagging(BaseModel):
    datalake: Datalake


class DataAutoTaggingConfig(OverrideOutputsMixin, BaseModel):
    job: JobAutoTag
    auth: Auth
    run: Run = Run()
    input: InputDataAutoTagging
    output: OutputDataAutoTagging
    run_parameters: AutoTagRunParams
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── MODEL_PROCESS ────────────────────────────────────────────────────────


class InputModelProcess(BaseModel):
    model_version: ModelVersion


class ModelProcessConfig(OverrideOutputsMixin, BaseModel):
    job: JobModelProcess
    auth: Auth
    run: Run = Run()
    input: InputModelProcess
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── TRAINING (NEW input/output shape) ────────────────────────────────────────


class InputTraining(BaseModel):
    train_dataset_version: DatasetVersion | None = None
    test_dataset_version: DatasetVersion | None = None
    validation_dataset_version: DatasetVersion | None = None
    model_version: ModelVersion | None = None


class OutputTraining(BaseModel):
    experiment: Experiment


class TrainingConfig(OverrideOutputsMixin, BaseModel):
    job: JobTraining
    auth: Auth
    run: Run = Run()
    input: InputTraining = Field(default_factory=InputTraining)
    output: OutputTraining
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    augmentations_parameters: dict[str, Any] = Field(default_factory=dict)
    export_parameters: dict[str, Any] = Field(default_factory=dict)
