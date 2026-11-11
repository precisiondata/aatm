"""A stand-in for Chroma Cloud, backed by a local persistent database.

It lives in its own small module so the writer process, which unpickles the
connection, can import it cheaply.
"""

from dataclasses import dataclass

from aatm.vector_db import ChromaConnection


@dataclass(frozen=True, slots=True)
class FakeCloudConnection(ChromaConnection):
    """Behaves like a cloud connection but stores data in ``path``."""

    def client(self):
        import chromadb

        return chromadb.PersistentClient(path=self.path)

    def describe(self) -> str:
        return f"fake cloud at '{self.path}'"


def fake_cloud(path) -> FakeCloudConnection:
    return FakeCloudConnection(mode="cloud", path=str(path), api_key="secret-key")
