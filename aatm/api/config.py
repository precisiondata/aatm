from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
import yaml


class APIConfig(BaseModel):
    DEFAULT_PATH: Path = Path(".aatm/api_config.yaml")
    host: str
    port: str
    batch_size: int
    rate_limit: Optional[int] = None
    workers: Optional[int] = None

    model_config = ConfigDict(extra="allow")

    def save_to_disk(self, path: str | Path = DEFAULT_PATH) -> None:
        if isinstance(path, str):
            path = Path(path)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json")))

    @classmethod
    def load_from_disk(cls, path: str | Path = DEFAULT_PATH) -> "APIConfig":
        if isinstance(path, str):
            path = Path(path)
        return cls(**yaml.safe_load(path.read_text()))
