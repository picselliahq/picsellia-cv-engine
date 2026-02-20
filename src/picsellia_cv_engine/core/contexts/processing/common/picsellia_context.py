import os
from typing import Any, Generic, TypeVar

import picsellia  # type: ignore

from picsellia_cv_engine.core.contexts import PicselliaContext
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaProcessingContext(PicselliaContext, Generic[TParameters]):
    """
    Base context for Picsellia processing jobs.
    """

    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        job_id: str | None = None,
        use_id: bool | None = True,
        working_dir: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_token=api_token,
            host=host,
            organization_id=organization_id,
            organization_name=organization_name,
            working_dir=working_dir,
        )

        self.job_id = job_id or os.environ.get("job_id")
        if not self.job_id:
            raise ValueError(
                "Job ID not provided. Please provide it as an argument or set the 'job_id' environment variable."
            )

        self.job: picsellia.Job = self._initialize_job()
        self.job_info: dict[str, Any] = self.job.sync()

        self.job_context: dict[str, Any] = self._initialize_job_context()

        print(f"job_info: {self.job_info}")

        print(f"job_context: {self.job_context}")

        self.parameters: dict[str, Any] = self.job_context.get("parameters", {}) or {}
        self.inputs: dict[str, Any] = self.job_context.get("inputs", {}) or {}
        self.payload_presigned_url: str | None = self.job_context.get(
            "payload_presigned_url"
        )

        self.use_id = use_id

        self.processing_parameters: TParameters = processing_parameters_cls(
            log_data=self.parameters
        )

        self._load_inputs(**kwargs)

    @property
    def working_dir(self) -> str:
        if self._working_dir_override:
            return self._working_dir_override
        return os.path.join(os.getcwd(), f"job_{self.job_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_id": self.job_id,
            },
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
        }

    def _initialize_job(self) -> picsellia.Job:
        return self.client.get_job_by_id(self.job_id)

    def _initialize_job_context(self) -> dict[str, Any]:
        try:
            return self.job_info["processing_job"]
        except KeyError as e:
            raise ValueError(
                f"Job context key 'processing_job' not found in job.sync(). "
                f"Available keys: {list(self.job_info.keys())}"
            ) from e

    def _load_inputs(self, **kwargs: Any) -> None:
        """
        Hook for subclasses to load inputs. Must not return anything.
        """
        return
