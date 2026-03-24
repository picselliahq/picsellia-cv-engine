from typing import Any, Generic, TypeVar

from picsellia_cv_engine.core.contexts import PicselliaContext
from picsellia_cv_engine.core.parameters import Parameters

TParameters = TypeVar("TParameters", bound=Parameters)


class PicselliaLocalProcessingContext(PicselliaContext, Generic[TParameters]):
    """
    Base context for Picsellia processing jobs.
    """

    def __init__(
        self,
        processing_parameters_cls: type[TParameters],
        parameters_dict: dict[str, Any] | None = None,
        api_token: str | None = None,
        host: str | None = None,
        organization_id: str | None = None,
        organization_name: str | None = None,
        target_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        use_id: bool | None = True,
        working_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(
            api_token=api_token,
            host=host,
            organization_id=organization_id,
            organization_name=organization_name,
            working_dir=working_dir,
        )

        self.target_id = target_id
        self.parameters: dict[str, Any] = (
            parameters_dict if parameters_dict is not None else {}
        )
        print("#### Inputs passed to local picsellia context", inputs)
        self.inputs = inputs

        self.use_id = use_id

        self.processing_parameters: TParameters = processing_parameters_cls(
            log_data=self.parameters
        )

        # TODO: remove this after full deprecation of legacy processing jobs
        self._load_legacy_inputs(**kwargs)

    @property
    def working_dir(self) -> str:
        return self._working_dir_override

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
            },
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
            "inputs": self.inputs,
        }

    def _load_legacy_inputs(self, **kwargs) -> None:
        """
        Hook for subclasses to load legacy inputs. Must not return anything.
        """
        return
