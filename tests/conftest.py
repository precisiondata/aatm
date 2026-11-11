import pytest

from aatm.vector_db import (
    CHROMA_API_KEY_ENV_VAR,
    CHROMA_CLOUD_HOST_ENV_VAR,
    CHROMA_DATABASE_ENV_VAR,
    CHROMA_TENANT_ENV_VAR,
    PINECONE_API_KEY_ENV_VAR,
    PINECONE_CLOUD_ENV_VAR,
    PINECONE_INDEX_PREFIX_ENV_VAR,
    PINECONE_REGION_ENV_VAR,
    VECTOR_DB_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _local_vector_db_by_default(monkeypatch):
    """Keep a developer's Chroma Cloud and Pinecone settings out of the tests.

    The Pinecone integration test reads ``PINECONE_API_KEY`` at import time,
    before this runs.
    """
    for name in (
        VECTOR_DB_ENV_VAR,
        CHROMA_API_KEY_ENV_VAR,
        CHROMA_TENANT_ENV_VAR,
        CHROMA_DATABASE_ENV_VAR,
        CHROMA_CLOUD_HOST_ENV_VAR,
        PINECONE_API_KEY_ENV_VAR,
        PINECONE_INDEX_PREFIX_ENV_VAR,
        PINECONE_CLOUD_ENV_VAR,
        PINECONE_REGION_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
