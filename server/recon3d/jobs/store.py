"""On-disk job store: <data_dir>/jobs/<job_id>/{capture.zip, status.json, scene.splat, scene.ply, log.txt}."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024**3
CHUNK = 1024 * 1024
ACTIVE_STATES = ("queued", "training", "exporting")
RESTART_MESSAGE = "服务重启，任务中断，请重新上传"
_JOB_ID = re.compile(r"[0-9a-f]{12}")


class UploadTooLargeError(Exception):
    def __init__(self, max_bytes: int):
        super().__init__(f"upload exceeds {max_bytes} bytes")
        self.max_bytes = max_bytes


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    progress: float
    message: str
    created_at: float


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

    def _commit(self, job_id: str, size: int) -> None:
        self._write(JobStatus(job_id, "queued", 0.0, "排队中", time.time()))
        self.log(job_id, f"created ({size} bytes), state=queued")
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
        """Jobs left active by a previous process can never finish; mark them failed."""
        for d in sorted(self.jobs_dir.iterdir()):
            s = self.get(d.name)
            if s is not None and s.state in ACTIVE_STATES:
                self.update(s.job_id, state="failed", message=RESTART_MESSAGE)

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
