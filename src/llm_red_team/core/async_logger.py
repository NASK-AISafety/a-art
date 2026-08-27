"""
A-ART Async Logger: Non-Blocking JSONL Logging for High-Throughput Pipelines.

Uses an ``asyncio.Queue`` consumer pattern to decouple write latency from the
inference hot-path.  The single consumer task batches records, acquires a
``filelock`` (for Slurm cross-process safety), and performs atomic
append + fsync.

Design rationale (ADR-001):
    - Queue decouples write latency from caller (fire-and-forget)
    - Consumer batches writes, reducing fsync calls
    - filelock contention is limited to one consumer task
    - ``await queue.join()`` ensures all records are flushed on shutdown

Usage::

    async with AsyncJSONLLogger("attacks.jsonl") as log:
        await log.log(entry)          # non-blocking enqueue
        await log.log_batch(entries)   # bulk enqueue
    # __aexit__ drains the queue and flushes to disk
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Union

import aiofiles
from pydantic import BaseModel

try:
    from filelock import FileLock

    _HAS_FILELOCK = True
except ImportError:
    FileLock = None  # type: ignore[assignment, misc]
    _HAS_FILELOCK = False

logger = logging.getLogger(__name__)

# Type alias — accepts any Pydantic model with a ``to_jsonl()`` method,
# or a plain dict (which will be serialised via json.dumps).
LogRecord = Union[BaseModel, dict[str, Any]]


class AsyncJSONLLogger:
    """
    Non-blocking, process-safe JSONL logger backed by ``asyncio.Queue``.

    Parameters
    ----------
    filepath:
        Path to the JSONL output file.  Parent directories are created
        automatically.
    buffer_size:
        Maximum number of records the consumer accumulates before writing
        to disk.  ``1`` means immediate write per record (lowest latency,
        highest fsync overhead).  ``10`` is a good default for HPC.
    use_filelock:
        Enable OS-level file locking via ``filelock`` for cross-process
        safety on Slurm clusters.
    flush_interval:
        Maximum seconds the consumer waits before flushing a partial
        buffer.  Ensures data is persisted even under low throughput.
    """

    def __init__(
        self,
        filepath: str | Path,
        buffer_size: int = 10,
        use_filelock: bool = True,
        flush_interval: float = 2.0,
    ) -> None:
        self._filepath = Path(filepath)
        self._buffer_size = max(1, buffer_size)
        self._flush_interval = flush_interval
        self._use_filelock = use_filelock and _HAS_FILELOCK

        # Ensure output directory exists
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        # File lock (cross-process safety)
        self._file_lock: FileLock | None = None  # type: ignore[assignment]
        if self._use_filelock and FileLock is not None:
            self._file_lock = FileLock(str(self._filepath) + ".lock")

        # Internal state
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._consumer_task: asyncio.Task[None] | None = None
        self._entries_written: int = 0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def filepath(self) -> Path:
        return self._filepath

    @property
    def entries_written(self) -> int:
        return self._entries_written

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the background consumer task."""
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(
            self._consumer_loop(), name="async-jsonl-consumer"
        )
        logger.debug("AsyncJSONLLogger consumer started for %s", self._filepath)

    async def stop(self) -> None:
        """Signal the consumer to drain and stop, then wait for completion."""
        if not self._running:
            return
        self._running = False
        # Sentinel to unblock the consumer if it's waiting on queue.get()
        await self._queue.put(None)
        if self._consumer_task is not None:
            await self._consumer_task
            self._consumer_task = None
        logger.debug(
            "AsyncJSONLLogger stopped — %d entries written to %s",
            self._entries_written,
            self._filepath,
        )

    async def log(self, record: LogRecord) -> None:
        """
        Enqueue a single record for async writing.

        Accepts a Pydantic model (calls ``model_dump_json()``) or a plain
        dict (serialised via ``json.dumps``).
        """
        line = self._serialise(record)
        await self._queue.put(line)

    async def log_batch(self, records: list[LogRecord]) -> None:
        """Enqueue multiple records at once."""
        for record in records:
            line = self._serialise(record)
            await self._queue.put(line)

    # ------------------------------------------------------------------
    # Async Context Manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncJSONLLogger:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Consumer Loop
    # ------------------------------------------------------------------

    async def _consumer_loop(self) -> None:
        """
        Background task: drain the queue in batches and write to disk.

        Flushes when *either* ``buffer_size`` records are collected *or*
        ``flush_interval`` seconds elapse — whichever comes first.
        """
        buffer: list[str] = []

        while True:
            # --- Drain phase: collect up to buffer_size records ---
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            except asyncio.TimeoutError:
                # Flush interval elapsed — write whatever we have
                if buffer:
                    await self._write_lines(buffer)
                    buffer.clear()
                continue

            if item is None:
                # Sentinel received — drain remaining items and exit
                while not self._queue.empty():
                    remaining = self._queue.get_nowait()
                    if remaining is not None:
                        buffer.append(remaining)
                    self._queue.task_done()
                if buffer:
                    await self._write_lines(buffer)
                    buffer.clear()
                self._queue.task_done()
                break

            buffer.append(item)
            self._queue.task_done()

            if len(buffer) >= self._buffer_size:
                await self._write_lines(buffer)
                buffer.clear()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    async def _write_lines(self, lines: list[str]) -> None:
        """
        Atomically append *lines* to the JSONL file.

        Uses ``aiofiles`` for non-blocking I/O and ``filelock`` for
        cross-process safety.
        """
        content = "\n".join(lines) + "\n"

        if self._file_lock is not None:
            # filelock is synchronous — run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._sync_write_with_lock, content)
        else:
            async with aiofiles.open(self._filepath, "a", encoding="utf-8") as f:
                await f.write(content)
                await f.flush()
                # fsync via executor (aiofiles doesn't expose it directly)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._fsync, f)

        self._entries_written += len(lines)

    def _sync_write_with_lock(self, content: str) -> None:
        """Synchronous write under filelock — called from executor."""
        assert self._file_lock is not None
        with self._file_lock:
            with open(self._filepath, "a", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def _fsync(f: Any) -> None:
        """fsync the underlying file descriptor."""
        try:
            os.fsync(f.fileno())
        except (AttributeError, OSError):
            pass  # aiofiles wrapper may not expose fileno in all backends

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(record: LogRecord) -> str:
        """
        Convert a record to a single JSON line.

        Validation (Issue J):
            - If ``record`` is already a Pydantic ``BaseModel`` it is serialised
              directly via ``model_dump_json()`` — assumed pre-validated.
            - If ``record`` is a plain ``dict`` it is validated through
              ``LogEntry`` (the canonical hot-path schema).  Validation errors
              are logged as ``WARNING`` and the raw dict is written unchanged so
              that no data is ever silently dropped.  This gives early warning
              about schema drift (e.g. a new field added to ``_build_log_entry``
              that has no corresponding ``LogEntry`` field) without killing the
              pipeline mid-run.
        """
        import json

        from llm_red_team.schemas.llm_red_team_schema import LogEntry

        if isinstance(record, BaseModel):
            return record.model_dump_json()

        # Plain dict — validate through LogEntry before serialising.
        try:
            validated = LogEntry.model_validate(record)
            return validated.model_dump_json()
        except Exception as exc:  # pydantic.ValidationError or anything else
            logger.warning(
                "LogEntry validation failed — writing raw dict. Error: %s | Fields present: %s",
                exc,
                list(record.keys()) if isinstance(record, dict) else type(record),
            )
            return json.dumps(record, default=str)
