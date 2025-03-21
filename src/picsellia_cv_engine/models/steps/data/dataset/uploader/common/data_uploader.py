from picsellia import Data, Datalake
from picsellia.services.error_manager import ErrorManager


class DataUploader:
    def _upload_data_with_error_manager(
        self,
        datalake: Datalake,
        images_to_upload: list[str],
        data_tags: list[str] | None = None,
    ) -> tuple[list[Data], list[str]]:
        error_manager = ErrorManager()
        data = datalake.upload_data(
            filepaths=images_to_upload, tags=data_tags, error_manager=error_manager
        )

        uploaded_data = (
            [data]
            if isinstance(data, Data)
            else [d for d in data if isinstance(d, Data)]
        )
        error_paths = [error.path for error in error_manager.errors]
        return uploaded_data, error_paths

    def _upload_images_to_datalake(
        self,
        datalake: Datalake,
        images_to_upload: list[str],
        data_tags: list[str] | None = None,
        max_retries: int = 5,
    ) -> list[Data]:
        all_uploaded_data = []
        uploaded_data, error_paths = self._upload_data_with_error_manager(
            datalake=datalake, images_to_upload=images_to_upload, data_tags=data_tags
        )
        all_uploaded_data.extend(uploaded_data)
        retry_count = 0

        while error_paths and retry_count < max_retries:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                datalake=datalake, images_to_upload=error_paths, data_tags=data_tags
            )
            all_uploaded_data.extend(uploaded_data)
            retry_count += 1
        if error_paths:
            raise Exception(
                f"Failed to upload the following images: {error_paths} after {max_retries} retries."
            )
        return all_uploaded_data
