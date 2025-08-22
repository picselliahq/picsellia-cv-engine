from typing import Any, Literal

from pydantic import BaseModel, Field


class Auth(BaseModel):
    organization_name: str
    host: str | None = None


class Run(BaseModel):
    name: str | None = None
    working_dir: str | None = None
    mode: Literal["local", "picsellia"] | None = None


class Experiment(BaseModel):
    id: str
    name: str | None = None
    project_name: str | None = None
    url: str | None = None


class ModelVersion(BaseModel):
    id: str
    name: str | None = None
    origin_name: str | None = None
    url: str | None = None


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


class JobTraining(BaseModel):
    type: Literal["TRAINING"]


class JobPreAnn(BaseModel):
    type: Literal["PRE_ANNOTATION"]


class JobDSVCreate(BaseModel):
    type: Literal["DATASET_VERSION_CREATION"]


class JobAutoTag(BaseModel):
    type: Literal["DATA_AUTO_TAGGING"]


class AutoTagRunParams(BaseModel):
    offset: int = 0
    limit: int = 100


class TrainingConfig(BaseModel):
    job: JobTraining
    auth: Auth
    run: Run = Run()
    experiment: Experiment
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    augmentations_parameters: dict[str, Any] = Field(default_factory=dict)
    export_parameters: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionCreationConfig(BaseModel):
    job: JobDSVCreate
    auth: Auth
    run: Run = Run()
    input: dict[str, DatasetVersion]
    output: dict[str, DatasetVersion]
    parameters: dict[str, Any] = Field(default_factory=dict)


class PreAnnotationConfig(BaseModel):
    job: JobPreAnn
    auth: Auth
    run: Run = Run()
    input: dict[str, DatasetVersion | ModelVersion]
    parameters: dict[str, Any] = Field(default_factory=dict)


class DataAutoTaggingConfig(BaseModel):
    job: JobAutoTag
    auth: Auth
    run: Run = Run()
    input: dict[str, Datalake | ModelVersion]
    output: dict[str, Datalake]
    run_parameters: AutoTagRunParams
    parameters: dict[str, Any] = Field(default_factory=dict)
