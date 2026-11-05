"""Background thread that runs queued jobs one at a time."""
from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from recon3d.capture.format import CaptureFormatError
from recon3d.jobs.store import JobStore

Pipeline = Callable[[Path, Callable[[float], None]], None]

OOM_MESSAGE = "显存不足，请降低分辨率或改用服务器"
EXPORT_START = 0.9


def is_oom(e: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError (or a subclass), matched by name so this module never
    imports torch; or a RuntimeError such as "CUDA error: out of memory" / CUBLAS alloc."""
    if any(cls.__name__ == "OutOfMemoryError" for cls in type(e).__mro__):
        return True
    return "out of memory" in str(e).lower()


def failure_message(e: BaseException, job_id: str) -> str:
    """User-facing (Chinese) failure text. Never includes exception text; details go to log.txt."""
    if isinstance(e, CaptureFormatError):
        return e.message_zh
    if is_oom(e):
        return OOM_MESSAGE
    return f"重建失败，请重新拍摄后上传；如仍失败，把服务端 data/jobs/{job_id}/log.txt 发给开发者"


class Worker:
    def __init__(self, store: JobStore, pipeline: Pipeline):
        self.store = store
        self.pipeline = pipeline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="recon3d-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop after the current job (if any). Returns once the thread exits or timeout passes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.store.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            except Exception:  # e.g. disk error writing status; keep serving later jobs
                traceback.print_exc()

    def _run(self, job_id: str) -> None:
        store = self.store
        if store.get(job_id) is None:
            return  # deleted while queued (DELETE /jobs/{id}); nothing left to run
        store.update(job_id, state="training", progress=0.0, message="训练中")

        def on_progress(p: float) -> None:
            p = min(max(float(p), 0.0), 1.0)
            if p >= EXPORT_START:
                store.update(job_id, state="exporting", progress=p, message="导出中")
            else:
                store.update(job_id, state="training", progress=p, message="训练中")

        try:
            self.pipeline(store.job_dir(job_id), on_progress)
        except Exception as e:
            store.log(job_id, "pipeline failed:\n" + traceback.format_exc().rstrip())
            store.update(job_id, state="failed", message=failure_message(e, job_id))
        else:
            store.update(job_id, state="done", progress=1.0, message="完成")
