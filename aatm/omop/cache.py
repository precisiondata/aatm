"""A persistent cache of coding results, shared across batches and runs.

The cache is off unless you turn it on. Pass ``cache=True`` (the default file,
``.aatm/coding_cache.db``), a path or a `CodingCache` to `OmopTableBuilder` or
`OmopExtractionPipeline`:

```python
from aatm.omop import AgenticCoder, CodingCache, OmopExtractionPipeline

cache = CodingCache(max_entries=500_000)  # .aatm/coding_cache.db
pipeline = OmopExtractionPipeline(
    "lab_results",
    model_id="openai/gpt-5-mini",
    mapper=AgenticCoder("openai/gpt-5-mini", retriever="text-embedding-3-small"),
    cache=cache,
)
tables = await pipeline(notes)
tables.cache_stats  # hits, misses and hit rate of this batch

cache.info()   # entries, size and coders
cache.clear()  # delete everything and shrink the file
```

or clear it from the command line with ``aatm cache clear``.

## What is reused

A result is only reused for **exactly** the input the coder would get again, so a
cached run gives the same results as an uncached one:

- Coders that read context (`aatm.omop.AgenticCoder`, or a `TerminologyMapper` with
  `aatm.concept_search.AgenticSelector`) are keyed on the full context: the text, the
  related values (e.g. the lab value and unit), the line of the note and the target
  column. The same text in another sentence is coded again.
- Other coders are keyed on the text, the domain and the vocabularies.

Every key also holds the **coder's identity**: its class, models, settings, index,
prompts and the aatm version. Changing any of these (or upgrading aatm) makes the
earlier entries unreachable; they are evicted first when the cache is full, or
removed with `CodingCache.clear`. Results that ended in an error are not stored.

The identity also holds a fingerprint of the **data** the coder reads: the vocabulary
releases in the vocabulary database (and its size), and the id and record count of a
ChromaDB collection. Rebuilding either from other data therefore starts from an empty
cache, and a warning names the old entries so they can be cleared. A retriever whose
data cannot be fingerprinted (not a ChromaDB collection) is named in a warning when the
cache is opened: clear the cache yourself after rebuilding its index.

## Size

``max_entries`` and ``max_bytes`` bound the cache. When a write goes over a limit, the
entries used least recently are deleted first. ``max_bytes`` counts the stored results;
the file itself is somewhat larger. SQLite reuses the space of deleted entries, and
`CodingCache.clear` shrinks the file.
"""

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Sequence

from pydantic import BaseModel

from aatm.logs import get_logger
from aatm.omop.coding import CodedText, uses_context

if TYPE_CHECKING:  # pragma: no cover
    from aatm.concept_search.agent import CodingContext

logger = get_logger(__name__)

DEFAULT_CODING_CACHE = Path(".aatm/coding_cache.db")
"""The cache file used by ``cache=True``."""

CACHE_FORMAT = 1
"""Version of the stored format and of the key; part of every coder identity."""

_SQL_CHUNK = 500  # keys per query, below SQLite's variable limit


class CacheStats(BaseModel):
    """Lookups of distinct texts in the cache.

    Attributes:
        hits: Distinct texts (in their context) found in the cache.
        misses: Distinct texts that were coded.
        stored: Results written to the cache; misses that ended in an error are not.
    """

    hits: int = 0
    misses: int = 0
    stored: int = 0

    @property
    def hit_rate(self) -> Optional[float]:
        """Hits over lookups, or ``None`` before any lookup."""
        total = self.hits + self.misses
        return self.hits / total if total else None

    def __sub__(self, other: "CacheStats") -> "CacheStats":
        return CacheStats(
            hits=self.hits - other.hits,
            misses=self.misses - other.misses,
            stored=self.stored - other.stored,
        )


class CoderInfo(BaseModel):
    """A coder identity in the cache and how many entries it holds."""

    coder: str
    identity: dict[str, Any]
    entries: int


class CacheInfo(BaseModel):
    """What a cache file holds.

    Attributes:
        path: The cache file.
        entries: Stored results.
        bytes: Size of the stored results, the measure ``max_bytes`` limits.
        file_bytes: Size of the file on disk.
        max_entries: The entry limit, if any.
        max_bytes: The size limit, if any.
        coders: The coder identities with entries, most entries first.
    """

    path: str
    entries: int
    bytes: int
    file_bytes: int
    max_entries: Optional[int]
    max_bytes: Optional[int]
    coders: list[CoderInfo]


class CodingCache:
    """Coding results stored in a SQLite file.

    The file can be shared by several processes; each operation opens its own
    connection.

    Args:
        path: The cache file. Created, with its directory, if missing.
        max_entries: Most results kept. ``None`` for no limit.
        max_bytes: Most bytes of results kept (the file is somewhat larger).
            ``None`` for no limit.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_CODING_CACHE,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ):
        for name, limit in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if limit is not None and limit < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {limit}")
        self.path = Path(path)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS coders (
                    coder TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    identity TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY,
                    coder TEXT NOT NULL,
                    text TEXT NOT NULL,
                    result TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS entries_last_used ON entries (last_used);
                CREATE INDEX IF NOT EXISTS entries_coder ON entries (coder);
                """
            )

    def __repr__(self) -> str:
        return (
            f"CodingCache({str(self.path)!r}, max_entries={self.max_entries}, "
            f"max_bytes={self.max_bytes})"
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _now(con: sqlite3.Connection) -> float:
        """The current time, after every stored use, so the eviction order is strict
        even when two operations fall in the same clock tick."""
        latest = con.execute("SELECT MAX(last_used) FROM entries").fetchone()[0]
        now = time.time()
        return now if latest is None or now > latest else latest + 1e-6

    def __len__(self) -> int:
        with closing(self._connect()) as con:
            return con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def register(
        self, coder: str, identity: dict[str, Any], config: str = ""
    ) -> list[CoderInfo]:
        """Record what a coder identity hash stands for, for `info`.

        Args:
            coder: The identity hash.
            identity: The identity it stands for.
            config: A hash of the identity without the data fingerprints.

        Returns:
            Other coders with the same configuration and stored entries: the same
            coder, before its vocabulary or index changed.
        """
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT OR IGNORE INTO coders VALUES (?, ?, ?, ?)",
                (coder, config, _canonical(identity), time.time()),
            )
            if not config:
                return []
            rows = con.execute(
                "SELECT c.coder, c.identity, COUNT(e.key) FROM coders c "
                "JOIN entries e ON e.coder = c.coder "
                "WHERE c.config = ? AND c.coder != ? GROUP BY c.coder",
                (config, coder),
            ).fetchall()
        return [
            CoderInfo(coder=c, identity=json.loads(i), entries=n) for c, i, n in rows
        ]

    def get_many(self, keys: Iterable[str]) -> dict[str, CodedText]:
        """The stored results of the keys found, marked as used now."""
        keys = list(dict.fromkeys(keys))
        found: dict[str, CodedText] = {}
        with closing(self._connect()) as con, con:
            now = self._now(con)
            for chunk in _chunks(keys, _SQL_CHUNK):
                marks = ",".join("?" * len(chunk))
                rows = con.execute(
                    f"SELECT key, result FROM entries WHERE key IN ({marks})", chunk
                ).fetchall()
                for key, result in rows:
                    coded = CodedText.model_validate_json(result)
                    coded.cached = True
                    found[key] = coded
                if rows:
                    hit = [key for key, _ in rows]
                    con.execute(
                        f"UPDATE entries SET last_used = ? WHERE key IN "
                        f"({','.join('?' * len(hit))})",
                        [now, *hit],
                    )
        return found

    def put_many(self, coder: str, items: Sequence[tuple[str, CodedText]]) -> None:
        """Store results, then evict the least recently used beyond the limits."""
        if not items:
            return
        with closing(self._connect()) as con, con:
            now = self._now(con)
            rows = []
            for key, coded in items:
                result = coded.model_dump_json(exclude={"cached"})
                size = len(result.encode()) + len(coded.text.encode())
                rows.append((key, coder, coded.text, result, size, now, now))
            con.executemany(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)", rows
            )
            self._evict(con)

    def _evict(self, con: sqlite3.Connection) -> None:
        evicted = 0
        if self.max_entries is not None:
            count = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            if count > self.max_entries:
                evicted += con.execute(
                    "DELETE FROM entries WHERE key IN (SELECT key FROM entries "
                    "ORDER BY last_used, rowid LIMIT ?)",
                    (count - self.max_entries,),
                ).rowcount
        if self.max_bytes is not None:
            total = con.execute("SELECT COALESCE(SUM(size), 0) FROM entries").fetchone()
            excess = total[0] - self.max_bytes
            if excess > 0:
                doomed = []
                for key, size in con.execute(
                    "SELECT key, size FROM entries ORDER BY last_used, rowid"
                ):
                    doomed.append(key)
                    excess -= size
                    if excess <= 0:
                        break
                for chunk in _chunks(doomed, _SQL_CHUNK):
                    evicted += con.execute(
                        f"DELETE FROM entries WHERE key IN ({','.join('?' * len(chunk))})",
                        chunk,
                    ).rowcount
        if evicted:
            logger.info(f"Coding cache: evicted {evicted} least recently used entries")

    def clear(self, coder: Optional[str] = None) -> int:
        """Delete entries and shrink the file.

        Args:
            coder: Only delete this coder identity's entries (a hash from `info`, or
                `CachedCoder.coder_hash`). ``None`` deletes everything.

        Returns:
            The number of entries deleted.
        """
        with closing(self._connect()) as con:
            with con:
                if coder is None:
                    removed = con.execute("DELETE FROM entries").rowcount
                    con.execute("DELETE FROM coders")
                else:
                    removed = con.execute(
                        "DELETE FROM entries WHERE coder = ?", (coder,)
                    ).rowcount
                    con.execute("DELETE FROM coders WHERE coder = ?", (coder,))
            con.execute("VACUUM")
        return removed

    def info(self) -> CacheInfo:
        """Entries, size, limits and coder identities of the cache."""
        with closing(self._connect()) as con:
            entries, size = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM entries"
            ).fetchone()
            rows = con.execute(
                "SELECT e.coder, COALESCE(c.identity, '{}'), COUNT(*) FROM entries e "
                "LEFT JOIN coders c ON c.coder = e.coder GROUP BY e.coder "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
        return CacheInfo(
            path=str(self.path),
            entries=entries,
            bytes=size,
            file_bytes=self.path.stat().st_size if self.path.exists() else 0,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
            coders=[
                CoderInfo(coder=c, identity=json.loads(i), entries=n)
                for c, i, n in rows
            ],
        )


class CachedCoder:
    """A `aatm.omop.coding.TextCoder` that reuses results stored in a `CodingCache`.

    Texts not in the cache are coded by the wrapped coder in chunks, most frequent
    first when the caller orders them so (`OmopTableBuilder` does), and each chunk is
    stored as soon as it is coded, so a run that stops partway keeps what it paid for.

    Args:
        coder: The coder to wrap. It must describe itself with ``cache_identity()``
            (the built-in coders do), or `identity` must be given.
        cache: A `CodingCache`, or the path of its file.
        identity: Everything that affects the coder's results (models, settings,
            prompts, index), as JSON-compatible values. Overrides
            ``coder.cache_identity()``.
        chunk_size: Texts coded between two writes to the cache.
    """

    def __init__(
        self,
        coder: Any,
        cache: "CodingCache | str | Path" = DEFAULT_CODING_CACHE,
        identity: Any = None,
        chunk_size: int = 64,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if isinstance(coder, CachedCoder):
            raise ValueError("The coder is already cached")
        if identity is None:
            describe = getattr(coder, "cache_identity", None)
            if not callable(describe):
                raise TypeError(
                    f"{type(coder).__name__} has no cache_identity(): pass "
                    "identity=... describing everything that affects its results "
                    "(models, settings, prompts, index)"
                )
            identity = describe()
        self.coder = coder
        self.cache = cache if isinstance(cache, CodingCache) else CodingCache(cache)
        self.identity = {
            "identity": identity,
            "aatm": _aatm_version(),
            "format": CACHE_FORMAT,
        }
        self.coder_hash = _digest(self.identity)
        self.chunk_size = chunk_size
        self.stats = CacheStats()
        stale = self.cache.register(
            self.coder_hash, self.identity, _digest(_without_content(self.identity))
        )
        for old in stale:
            logger.warning(
                f"Coding cache {self.cache.path}: the vocabulary or search index of "
                f"this coder changed since {old.entries} results were cached for it "
                f"(coder {old.coder}). They will not be reused. To free their space: "
                f"aatm cache clear --coder {old.coder}"
            )
        self.unchecked = unchecked_data(identity)
        if self.unchecked:
            logger.warning(
                f"Coding cache {self.cache.path}: changes to the content of "
                f"{', '.join(self.unchecked)} cannot be detected. After rebuilding "
                "it, run `aatm cache clear`, or cached results from the old data "
                "will be reused."
            )

    @property
    def uses_context(self) -> bool:
        return uses_context(self.coder)

    def key(
        self,
        text: str,
        domain: Optional[str],
        vocabularies: Optional[Sequence[str]],
        context_key: str = "",
    ) -> str:
        """The cache key of a text: the coder identity, the scope and the context."""
        payload = [
            self.coder_hash,
            text,
            domain,
            sorted(vocabularies or []),
            context_key,
        ]
        return hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence["CodingContext"]] = None,
    ) -> list[CodedText]:
        if contexts is not None and len(contexts) != len(texts):
            raise ValueError(f"Got {len(contexts)} contexts for {len(texts)} texts")
        with_context = self.uses_context and contexts is not None
        context_keys = (
            [c.cache_key() for c in contexts]  # type: ignore[union-attr]
            if with_context
            else [""] * len(texts)
        )
        keys = [
            self.key(t, domain, vocabularies, ck) for t, ck in zip(texts, context_keys)
        ]
        found = self.cache.get_many(keys)
        # Distinct misses, in the caller's order.
        missing = [k for k in dict.fromkeys(keys) if k not in found]
        self.stats.hits += len(found)
        self.stats.misses += len(missing)
        position = {k: i for i, k in reversed(list(enumerate(keys)))}

        for chunk in _chunks(missing, self.chunk_size):
            indices = [position[k] for k in chunk]
            kwargs: dict[str, Any] = {}
            if with_context:
                kwargs["contexts"] = [contexts[i] for i in indices]  # type: ignore[index]
            results = self.coder.code(
                [texts[i] for i in indices], domain, vocabularies, **kwargs
            )
            if len(results) != len(chunk):
                raise ValueError(
                    f"{type(self.coder).__name__} returned {len(results)} results "
                    f"for {len(chunk)} texts"
                )
            to_store = []
            for key, result in zip(chunk, results):
                found[key] = result
                if result.error is None:
                    to_store.append((key, result))
            self.cache.put_many(self.coder_hash, to_store)
            self.stats.stored += len(to_store)

        if keys:
            logger.info(
                f"Coding cache: {len(keys) - len(missing)} of {len(keys)} texts found, "
                f"{len(missing)} coded"
            )
        return [found[k] for k in keys]


# --- coder identities -----------------------------------------------------------

# Attributes of pipeline components that change what they return. Values that are
# not JSON-compatible (clients, loaded models) are left out.
_SETTING_ATTRS = (
    "model_id",
    "model",
    "model_name",
    "collection_name",
    "path",
    "top_k",
    "where",
    "filter_strategy",
    "overfetch_factor",
    "min_overfetch",
    "max_overfetch",
    "max_tool_calls",
    "domain",
    "vocabularies",
    "limit_to",
    "settings",
)
# Generation settings (`aatm.llm.LLMSettings`) left out of identities: changing them
# does not change the results, so it should not start a new cache.
_RESULT_NEUTRAL_SETTINGS = ("timeout",)
# Attributes holding nested components.
_COMPONENT_ATTRS = (
    "translator",
    "retriever",
    "reranker",
    "selector",
    "agent",
    "llm",
    "tools",
    "vocabulary",
    "embedding_function",
)


def describe_component(obj: Any, _depth: int = 0) -> Any:
    """A JSON-compatible description of a pipeline component, for cache identities.

    The class, the settings that change its results (models, index, filters, top k,
    generation settings) and its nested components (retriever, selector, agent...).
    The concept search agent's prompts are included as a fingerprint.
    """
    plain = _plain(obj)
    if plain is not _UNSET:
        return plain
    cls = type(obj)
    described: dict[str, Any] = {"class": f"{cls.__module__}.{cls.__qualname__}"}
    for name in _SETTING_ATTRS:
        value = _plain(getattr(obj, name, None))
        if name == "settings" and isinstance(value, dict):
            # Generation settings, minus those that do not change the answer.
            value = {
                k: v for k, v in value.items() if k not in _RESULT_NEUTRAL_SETTINGS
            }
        if value is not _UNSET and value is not None:
            described[name] = value
    base_url = getattr(getattr(obj, "client", None), "base_url", None)
    if isinstance(base_url, (str, Path)) or type(base_url).__name__ == "URL":
        # The server matters for OpenAI-compatible providers (Ollama, vLLM...).
        described["base_url"] = str(base_url)
    content = data_fingerprint(obj)
    if content is not None:
        described["content"] = content
    if _depth < 5:
        for name in _COMPONENT_ATTRS:
            child = getattr(obj, name, None)
            if child is None or name in described:
                continue
            if name == "vocabulary" and isinstance(child, (str, Path)):
                # A path to the vocabulary database (AgenticSelector).
                described[name] = {"path": Path(child).as_posix()}
                fingerprint = vocabulary_fingerprint(child)
                if fingerprint is not None:
                    described[name]["content"] = fingerprint
            else:
                described[name] = describe_component(child, _depth + 1)
    from aatm.concept_search.agent import ConceptSearchAgent
    from aatm.concept_search.selector import AgenticSelector

    # The selector's agent may not exist yet: its retriever is bound later.
    if isinstance(obj, (ConceptSearchAgent, AgenticSelector)):
        described["prompts"] = agent_prompt_fingerprint()
    return described


# --- content of data sources ------------------------------------------------------

# Components whose results depend on data they read: a change to that data must
# change the identity, or old results would be reused.
_DATA_ATTRS = ("retriever", "vocabulary")


def data_fingerprint(obj: Any) -> Optional[dict[str, Any]]:
    """What identifies the content a component reads, when it can be read cheaply.

    - A vocabulary database (`aatm.omop.vocabulary.SqliteConceptLookup`): see
      `vocabulary_fingerprint`.
    - A ChromaDB collection: its id, which changes when the collection is
      recreated, and its number of records.
    - Another vector database index (`aatm.retrievers.PineconeRetriever`): what
      its ``content_fingerprint()`` returns, i.e. its name, host, dimension and
      number of records.

    Returns ``None`` for components without such data, or whose data is unknown.
    """
    from aatm.omop.vocabulary import SqliteConceptLookup

    if isinstance(obj, SqliteConceptLookup):
        return vocabulary_fingerprint(obj.path)
    collection = getattr(obj, "collection", None)
    if collection is not None and hasattr(collection, "id"):
        try:
            return {"collection_id": str(collection.id), "count": collection.count()}
        except Exception as e:  # pragma: no cover - a broken client
            logger.warning(
                f"Could not read the collection of {type(obj).__name__}: {e}"
            )
        return None
    # Other vector databases, e.g. a Pinecone index: name, host, dimension, count.
    content_fingerprint = getattr(obj, "content_fingerprint", None)
    if callable(content_fingerprint):
        try:
            return content_fingerprint()
        except Exception as e:  # pragma: no cover - a broken client
            logger.warning(f"Could not read the index of {type(obj).__name__}: {e}")
    return None


def vocabulary_fingerprint(path: str | Path) -> Optional[dict[str, Any]]:
    """The vocabulary releases in a vocabulary database, and the file size.

    Rebuilding the database from another Athena download changes the
    ``vocabulary_version`` of its vocabularies (e.g. ``"LOINC 2.78"``); adding or
    removing vocabularies changes the list and the size. A database without a
    ``vocabulary`` table is identified by its size and modification time.
    """
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    try:
        with closing(
            sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        ) as con:
            versions = con.execute(
                "SELECT vocabulary_id, vocabulary_version FROM vocabulary "
                "ORDER BY vocabulary_id"
            ).fetchall()
    except sqlite3.Error:
        return {"bytes": stat.st_size, "modified": stat.st_mtime_ns}
    digest = hashlib.sha256(_canonical(versions).encode()).hexdigest()[:16]
    return {"versions": digest, "bytes": stat.st_size}


def unchecked_data(identity: Any, _path: str = "") -> list[str]:
    """Data components of an identity whose content is not fingerprinted."""
    found: list[str] = []
    if isinstance(identity, dict):
        for name, value in identity.items():
            where = f"{_path}.{name}" if _path else name
            if (
                name in _DATA_ATTRS
                and isinstance(value, dict)
                and "content" not in value
            ):
                found.append(f"{where} ({value.get('class', 'unknown')})")
            found.extend(unchecked_data(value, where))
    return found


def _without_content(identity: Any) -> Any:
    """The identity without data fingerprints: the coder's configuration alone."""
    if isinstance(identity, dict):
        return {k: _without_content(v) for k, v in identity.items() if k != "content"}
    if isinstance(identity, list):
        return [_without_content(v) for v in identity]
    return identity


def agent_prompt_fingerprint() -> str:
    """A hash of the concept search agent's prompts and tool descriptions."""
    from aatm.concept_search import agent

    text = _canonical([agent.SYSTEM_PROMPT, agent.ACTION_HELP])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


_UNSET = object()


def _plain(value: Any) -> Any:
    """The value as JSON-compatible data, or `_UNSET` when it is an object."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        items = {str(k): _plain(v) for k, v in value.items()}
        return items if _UNSET not in items.values() else _UNSET
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_plain(v) for v in value]
        if _UNSET in values:
            return _UNSET
        return (
            sorted(values, key=repr) if isinstance(value, (set, frozenset)) else values
        )
    return _UNSET


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:32]


def _aatm_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("aatm")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "unknown"


def _chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def as_cache(cache: Any) -> Optional[CodingCache]:
    """The `CodingCache` of a ``cache`` argument: ``None``/``False`` (off), ``True``
    (the default file), a path, or a `CodingCache`."""
    if cache is None or cache is False:
        return None
    if cache is True:
        return CodingCache()
    if isinstance(cache, CodingCache):
        return cache
    if isinstance(cache, (str, Path)):
        return CodingCache(cache)
    raise TypeError(
        f"cache must be None, a bool, a path or a CodingCache, not {type(cache).__name__}"
    )
