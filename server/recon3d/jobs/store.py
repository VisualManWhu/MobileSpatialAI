"""On-disk job store: <data_dir>/jobs/<job_id>/{capture.zip, status.json, scene.splat, scene.ply, log.txt}.

A resumable upload lives in the same directory before it becomes a job: capture.zip.part + upload.json
(size, sha256, received, created_at, updated_at) and no status.json, so list()/recover_after_restart()
ignore it until finish_resumable() renames the part and writes the queued status.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3
CHUNK = 1024 * 1024
ACTIVE_STATES = ("queued", "training", "exporting")
RESTART_MESSAGE = "服务重启，任务中断，请重新上传"
RESUMABLE_MAX_AGE_S = 24 * 3600
_JOB_ID = re.compile(r"[0-9a-f]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class UploadTooLargeError(Exception):
    def __init__(self, max_bytes: int):
        super().__init__(f"upload exceeds {max_bytes} bytes")
        self.max_bytes = max_bytes


class _ReceivedError(Exception):
    """Base for resumable-upload errors that carry the server's current byte count."""

    def __init__(self, received: int):
        super().__init__(f"received={received}")
        self.received = received


class OffsetMismatch(_ReceivedError):
    """Upload-Offset differs from what is on disk."""


class UploadBusy(_ReceivedError):
    """Another append for the same upload is in progress."""


class IncompleteUpload(_ReceivedError):
    """finish() before every byte arrived."""


class ChecksumMismatch(Exception):
    """The finished part's SHA-256 is not the one declared at begin; the upload is deleted."""


class JobActive(Exception):
    """delete() while the worker is training/exporting the job (its directory is being written)."""


@dataclass(frozen=True)
class ResumableInfo:
    upload_id: str
    size: int
    sha256: str
    received: int
    created_at: float
    updated_at: float


MODES = ("fast", "precise")


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    progress: float
    message: str
    created_at: float
    mode: str = "fast"  # "fast" | "precise"; absent in status.json written before precise mode


class Upload:
    """A capture.zip being written. write() enforces the store's size cap as bytes arrive."""

    def __init__(self, store: JobStore):
        self._store = store
        self.job_id = uuid.uuid4().hex[:12]
        self.dir = store.job_dir(self.job_id)
        self.dir.mkdir()
        self.size = 0
        self._f = open(self.dir / "capture.zip", "wb")

    def write(self, data: bytes) -> None:
        self.size += len(data)
        if self.size > self._store.max_upload_bytes:
            raise UploadTooLargeError(self._store.max_upload_bytes)
        self._f.write(data)

    def abort(self) -> None:
        """Discard the partial upload (idempotent)."""
        self._f.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def commit(self) -> str:
        """Finish the upload, write status queued and enqueue the job."""
        try:
            self._f.close()
            self._store._commit(self.job_id, self.size)
        except BaseException:
            self.abort()
            raise
        return self.job_id


class JobStore:
    def __init__(self, data_dir: Path, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES):
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes
        self.queue: queue.Queue[str] = queue.Queue()  # ids of jobs created by this process
        self._lock = threading.Lock()
        self._upload_locks: dict[str, threading.Lock] = {}

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def begin_upload(self) -> Upload:
        """Open a new job's capture.zip for incremental writing; finish with commit() or abort()."""
        return Upload(self)

    def create(self, stream: BinaryIO) -> str:
        """Copy a file-like to a new job in chunks, then enqueue. Raises UploadTooLargeError."""
        up = self.begin_upload()
        try:
            while chunk := stream.read(CHUNK):
                up.write(chunk)
        except BaseException:
            up.abort()
            raise
        return up.commit()

    def _commit(self, job_id: str, size: int, mode: str = "fast") -> None:
        self._write(JobStatus(job_id, "queued", 0.0, "排队中", time.time(), mode))
        self.log(job_id, f"created ({size} bytes), state=queued, mode={mode}")
        self.queue.put(job_id)

    def get(self, job_id: str) -> JobStatus | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        try:
            raw = json.loads((self.job_dir(job_id) / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return JobStatus(**raw)

    def list(self) -> list[JobStatus]:
        """Every job with a readable status.json, newest first (leftover dirs are skipped)."""
        found = [s for d in self.jobs_dir.iterdir() if (s := self.get(d.name)) is not None]
        return sorted(found, key=lambda s: s.created_at, reverse=True)

    def update(self, job_id: str, **fields) -> JobStatus:
        with self._lock:
            old = self.get(job_id)
            if old is None:
                raise KeyError(job_id)
            new = JobStatus(**{**asdict(old), **fields})
            self._write(new)
        if new.state != old.state:
            self.log(job_id, f"state={new.state} message={new.message}")
        return new

    def recover_after_restart(self) -> None:
        """Jobs left active by a previous process can never finish; mark them failed.

        Expired resumable uploads are removed and the surviving ones get `received` corrected from the
        part file (the previous process may have died between writing bytes and updating upload.json).
        """
        self.sweep_resumable()
        for d in sorted(self.jobs_dir.iterdir()):
            s = self.get(d.name)
            if s is not None and s.state in ACTIVE_STATES:
                self.update(s.job_id, state="failed", message=RESTART_MESSAGE)
            info = self.resumable(d.name)
            if info is not None:
                self._write_upload_meta(info, received=self._part_size(d.name))

    def delete(self, job_id: str) -> None:
        """Remove a job (done/failed/queued) or an in-progress resumable upload, directory and all.

        Raises KeyError (unknown or malformed id), JobActive (training/exporting: the pipeline is
        writing into the directory) or UploadBusy (a PUT holds the upload lock). A queued job's id
        stays in the worker queue; Worker._run skips ids whose status is gone.
        """
        if not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        d = self.job_dir(job_id)
        if not d.is_dir():
            raise KeyError(job_id)
        lock = self._upload_lock(job_id)
        if not lock.acquire(blocking=False):
            raise UploadBusy(self._part_size(job_id))
        try:
            with self._lock:  # update() holds it for its read-modify-write; no status write interleaves
                s = self.get(job_id)
                if s is not None and s.state in ("training", "exporting"):
                    raise JobActive(job_id)
                shutil.rmtree(d, ignore_errors=True)
                self._upload_locks.pop(job_id, None)
        finally:
            lock.release()

    # --- resumable uploads ---

    def begin_resumable(self, size: int, sha256: str) -> ResumableInfo:
        """Register an upload of `size` bytes with the given SHA-256; bytes arrive via append_resumable."""
        if size <= 0:
            raise ValueError("size must be positive")
        if not _SHA256.fullmatch(sha256):
            raise ValueError("sha256 must be 64 lowercase hex digits")
        if size > self.max_upload_bytes:
            raise UploadTooLargeError(self.max_upload_bytes)
        upload_id = uuid.uuid4().hex[:12]
        d = self.job_dir(upload_id)
        d.mkdir()
        (d / "capture.zip.part").touch()
        now = time.time()
        info = ResumableInfo(upload_id, size, sha256, 0, now, now)
        self._write_upload_meta(info)
        return info

    def resumable(self, upload_id: str) -> ResumableInfo | None:
        """The upload's metadata, or None when unknown or already finished.

        `received` is the part file's current size (the truth even while a PUT is streaming in);
        upload.json's copy is only a record for the status of a crashed process.
        """
        if not _JOB_ID.fullmatch(upload_id):
            return None
        try:
            raw = json.loads((self.job_dir(upload_id) / "upload.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return ResumableInfo(upload_id=upload_id, **{**raw, "received": self._part_size(upload_id)})

    def list_resumable(self) -> list[ResumableInfo]:
        """Every upload still being received, newest first (finished ones are jobs, see list())."""
        found = [i for d in self.jobs_dir.iterdir() if (i := self.resumable(d.name)) is not None]
        return sorted(found, key=lambda i: i.created_at, reverse=True)

    def append_resumable(self, upload_id: str, offset: int, chunks: Iterable[bytes]) -> int:
        """Write `chunks` at `offset` (which must equal the bytes already received); returns the new count.

        Bytes written before the iterable raises (a dropped connection) are kept and counted. Raises
        OffsetMismatch, UploadBusy, UploadTooLargeError (and deletes the upload) or KeyError (unknown id).
        """
        info = self.resumable(upload_id)
        if info is None:
            raise KeyError(upload_id)
        lock = self._upload_lock(upload_id)
        if not lock.acquire(blocking=False):
            raise UploadBusy(self._part_size(upload_id))
        try:
            received = self._part_size(upload_id)
            if offset != received:
                raise OffsetMismatch(received)
            too_large = False
            try:
                with open(self.job_dir(upload_id) / "capture.zip.part", "r+b") as f:
                    f.seek(offset)
                    for chunk in chunks:
                        if received + len(chunk) > info.size:
                            too_large = True
                            break
                        f.write(chunk)
                        f.flush()  # the part's size is the truth a concurrent caller reports
                        received += len(chunk)
                    os.fsync(f.fileno())
            finally:
                if too_large:
                    shutil.rmtree(self.job_dir(upload_id), ignore_errors=True)
                else:
                    self._write_upload_meta(info, received=received)
            if too_large:
                raise UploadTooLargeError(info.size)
            return received
        finally:
            lock.release()

    def finish_resumable(self, upload_id: str) -> str:
        """Verify size and SHA-256, turn the upload into a queued job and return its id (idempotent)."""
        info = self.resumable(upload_id)
        if info is None:
            if self.get(upload_id) is not None:
                return upload_id
            raise KeyError(upload_id)
        with self._upload_lock(upload_id):
            # Re-read under the lock: a finish that waited behind another one finds the job already made.
            info = self.resumable(upload_id)
            if info is None:
                if self.get(upload_id) is not None:
                    return upload_id
                raise KeyError(upload_id)
            d = self.job_dir(upload_id)
            part = d / "capture.zip.part"
            received = self._part_size(upload_id)
            if received != info.size:
                raise IncompleteUpload(received)
            h = hashlib.sha256()
            with open(part, "rb") as f:
                while block := f.read(CHUNK):
                    h.update(block)
            if h.hexdigest() != info.sha256:
                shutil.rmtree(d, ignore_errors=True)
                raise ChecksumMismatch()
            os.replace(part, d / "capture.zip")
            (d / "upload.json").unlink(missing_ok=True)
            self._commit(upload_id, info.size)
        return upload_id

    def sweep_resumable(self, now: float | None = None, max_age_s: float = RESUMABLE_MAX_AGE_S) -> int:
        """Delete uploads untouched for max_age_s (skipping ones being received); returns how many."""
        now = time.time() if now is None else now
        removed = 0
        for d in list(self.jobs_dir.iterdir()):
            info = self.resumable(d.name)
            if info is None or now - info.updated_at <= max_age_s:
                continue
            lock = self._upload_lock(d.name)
            if not lock.acquire(blocking=False):
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
            finally:
                lock.release()
        return removed

    def _upload_lock(self, upload_id: str) -> threading.Lock:
        with self._lock:
            return self._upload_locks.setdefault(upload_id, threading.Lock())

    def _part_size(self, upload_id: str) -> int:
        try:
            return (self.job_dir(upload_id) / "capture.zip.part").stat().st_size
        except OSError:
            return 0

    def _write_upload_meta(self, info: ResumableInfo, **fields) -> None:
        d = self.job_dir(info.upload_id)
        if not d.is_dir():
            return
        data = {**asdict(info), **fields, "updated_at": time.time()}
        data.pop("upload_id")
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".upload-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, d / "upload.json")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def log(self, job_id: str, text: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.job_dir(job_id) / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")

    def _write(self, status: JobStatus) -> None:
        d = self.job_dir(status.job_id)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".status-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(status), f, ensure_ascii=False)
            os.replace(tmp, d / "status.json")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
