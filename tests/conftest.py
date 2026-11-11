import pytest

from aatm.vector_db import (
    CHROMA_API_KEY_ENV_VAR,
    CHROMA_CLOUD_HOST_ENV_VAR,
    CHROMA_DATABASE_ENV_VAR,
    CHROMA_TENANT_ENV_VAR,
    VECTOR_DB_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _local_vector_db_by_default(monkeypatch):
    """Keep a developer's Chroma Cloud settings from leaking into the tests."""
    for name in (
        VECTOR_DB_ENV_VAR,
        CHROMA_API_KEY_ENV_VAR,
        CHROMA_TENANT_ENV_VAR,
        CHROMA_DATABASE_ENV_VAR,
        CHROMA_CLOUD_HOST_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
