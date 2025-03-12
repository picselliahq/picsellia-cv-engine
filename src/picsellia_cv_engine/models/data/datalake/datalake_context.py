import logging
import os
from uuid import UUID

from picsellia import Datalake

logger = logging.getLogger(__name__)


class DatalakeContext:
    """
    Manages the context for downloading data from a Datalake.

    This class allows you to configure and initialize the necessary paths for downloading
    data from a Datalake to a specified destination path. It handles initializing the access
    paths and downloading the data based on the specified IDs.
    """

    def __init__(
        self,
        datalake_name: str,
        datalake: Datalake,
        data_ids: list[UUID] | None = None,
        use_id: bool | None = True,
    ):
        """
        Initializes a DatalakeContext object.

        Args:
            datalake_name (str): The name of the Datalake.
            datalake (Datalake): The Datalake instance from which data will be downloaded.
            data_ids (list[UUID] | None, optional): A list of data IDs to download, or None to download all data.
            use_id (bool | None, optional): A flag indicating whether to use the IDs during data download. Default is True.
        """
        self.datalake_name: str = datalake_name
        """The name of the Datalake."""

        self.datalake: Datalake = datalake
        """The Datalake instance from which data will be downloaded."""

        self.data_ids: list[UUID] | None = data_ids
        """A list of data IDs to download, or None to download all data."""

        self.use_id: bool | None = use_id
        """A flag indicating whether to use the IDs during data download."""

        self.images_dir: str | None = None
        """The directory where the downloaded images will be saved."""

    def download_data(self, destination_dir: str) -> None:
        """
        Downloads data from the Datalake to the specified image directory.

        Args:
            destination_dir (str): The directory where the downloaded images will be saved.

        Raises:
            OSError: If the destination directory cannot be created.
        """
        os.makedirs(destination_dir, exist_ok=True)
        if self.data_ids:
            data = self.datalake.list_data(ids=self.data_ids)
            data.download(target_path=destination_dir, use_id=self.use_id)
            self.images_dir = destination_dir
