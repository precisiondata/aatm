"""Configuration model and persistence utilities for the AATM API.

This module defines the `APIConfig` model, which stores runtime configuration
for the API server and provides helper methods to save the configuration to
disk and load it back from a YAML file.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
import yaml


class APIConfig(BaseModel):
    """Configuration model for the AATM API server.

    This model stores the runtime settings used to serve the API, including host,
    port, batching behavior, optional rate limiting, and worker configuration. It
    also supports persistence to and from a YAML file.

    Attributes:
        DEFAULT_PATH: Default filesystem path used to save and load the API
            configuration.
        host: Host interface on which the API server listens.
        port: Port on which the API server listens.
        batch_size: Batch size used by the API processing pipeline.
        rate_limit: Optional maximum number of documents allowed per minute.
        workers: Optional number of worker processes.
    """

    DEFAULT_PATH: Path = Path(".aatm/api_config.yaml")
    host: str
    port: str
    batch_size: int
    rate_limit: Optional[int] = None
    workers: Optional[int] = None

    model_config = ConfigDict(extra="allow")

    def save_to_disk(self, path: str | Path = DEFAULT_PATH) -> None:
        """Save the API configuration to a YAML file on disk.

        Args:
            path: Destination path where the configuration should be written. If not
                provided, the default configuration path is used.

        Returns:
            None.
        """
        if isinstance(path, str):
            path = Path(path)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json")))

    @classmethod
    def load_from_disk(cls, path: str | Path = DEFAULT_PATH) -> "APIConfig":
        """Load the API configuration from a YAML file on disk.

        Args:
            path: Path to the YAML configuration file. If not provided, the default
                configuration path is used.

        Returns:
            An `APIConfig` instance initialized from the contents of the file.
        """
        if isinstance(path, str):
            path = Path(path)
        return cls(**yaml.safe_load(path.read_text()))
