from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class Auth(BaseModel):
    organization_name: str
    host: str | None = None


class Run(BaseModel):
    name: str | None = None
    working_dir: str | None = None
    mode: Literal["local", "picsellia"] | None = None


class ModelBlk(BaseModel):
    model_version_id: str


class JobPreAnn(BaseModel):
    type: Literal["PRE_ANNOTATION"]


class JobDSVCreate(BaseModel):
    type: Literal["DATASET_VERSION_CREATION"]


class JobAutoTag(BaseModel):
    type: Literal["DATA_AUTO_TAGGING"]


class IOInputDataset(BaseModel):
    input_dataset_version_id: str


class IOInputOutputDataset(BaseModel):
    input_dataset_version_id: str
    output_dataset_version_name: str


class IODatalake(BaseModel):
    input_datalake_id: str
    output_datalake_id: str


class AutoTagRunParams(BaseModel):
    offset: int = 0
    limit: int = 100


class PreAnnotationConfig(BaseModel):
    job: JobPreAnn
    auth: Auth
    run: Run = Run()
    io: IOInputDataset
    model: ModelBlk
    parameters: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionCreationConfig(BaseModel):
    job: JobDSVCreate
    auth: Auth
    run: Run = Run()
    io: IOInputOutputDataset
    parameters: dict[str, Any] = Field(default_factory=dict)


class DataAutoTaggingConfig(BaseModel):
    job: JobAutoTag
    auth: Auth
    run: Run = Run()
    io: IODatalake
    model: ModelBlk
    run_parameters: AutoTagRunParams
    parameters: dict[str, Any] = Field(default_factory=dict)


ProcessingConfig = Annotated[
    Union[PreAnnotationConfig, DatasetVersionCreationConfig, DataAutoTaggingConfig],
    Field(discriminator="job.type"),
]
