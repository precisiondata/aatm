import pytest

from aatm.vector_db import (
    CHROMA_API_KEY_ENV_VAR,
    CHROMA_CLOUD_HOST_ENV_VAR,
    CHROMA_DATABASE_ENV_VAR,
    CHROMA_TENANT_ENV_VAR,
    PG_DSN_ENV_VAR,
    PG_EF_SEARCH_ENV_VAR,
    PG_ITERATIVE_SCAN_ENV_VAR,
    PG_MAINTENANCE_WORK_MEM_ENV_VAR,
    PG_TABLE_PREFIX_ENV_VAR,
    PINECONE_API_KEY_ENV_VAR,
    PINECONE_CLOUD_ENV_VAR,
    PINECONE_INDEX_PREFIX_ENV_VAR,
    PINECONE_REGION_ENV_VAR,
    VECTOR_DB_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _local_vector_db_by_default(monkeypatch):
    """Keep a developer's Chroma Cloud, Pinecone and PostgreSQL settings out of the tests.

    The Pinecone and PostgreSQL integration tests read ``PINECONE_API_KEY`` and
    ``AATM_TEST_PG_DSN`` at import time, before this runs.
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
        PG_DSN_ENV_VAR,
        PG_TABLE_PREFIX_ENV_VAR,
        PG_EF_SEARCH_ENV_VAR,
        PG_ITERATIVE_SCAN_ENV_VAR,
        PG_MAINTENANCE_WORK_MEM_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _record_built_stores_in_tmp(monkeypatch, tmp_path_factory):
    """Keep the tests' records of built vector databases out of ``.aatm/config.yaml``."""
    from aatm import built_stores

    path = tmp_path_factory.mktemp("aatm") / "config.yaml"
    monkeypatch.setattr(built_stores, "CONFIG_PATH", path)
