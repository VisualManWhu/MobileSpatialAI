"""Background thread that runs queued jobs one at a time."""
from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from recon3d.capture.format import CaptureFormatError
from recon3d.jobs.pipeline import OnProgress
from recon3d.jobs.store import JobStore

# (job_dir, mode, on_progress) -> final "done" message to use instead of 完成, or None.
Pipeline = Callable[[Path, str, OnProgress], str | None]
# (job_dir, on_progress) -> number of objects found (phase 2 recognition over a done job).
ObjectsPipeline = Callable[[Path, OnProgress], int]

OOM_MESSAGE = "显存不足，请降低分辨率或改用服务器"
MODEL_DOWNLOAD_MESSAGE = "模型下载失败，请检查网络"
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


def objects_failure_message(e: BaseException, job_id: str) -> str:
    if is_oom(e):
        return OOM_MESSAGE
    if type(e).__name__ == "ModelDownloadFailed":
        return MODEL_DOWNLOAD_MESSAGE
    return f"识别失败，请重试；如仍失败，把服务端 data/jobs/{job_id}/log.txt 发给开发者"


class Worker:
    def __init__(self, store: JobStore, pipeline: Pipeline, objects_pipeline: ObjectsPipeline | None = None):
        self.store = store
        self.pipeline = pipeline
        self.objects_pipeline = objects_pipeline
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
                kind, job_id = self.store.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if kind == "objects":
                    self._run_objects(job_id)
                else:
                    self._run(job_id)
            except Exception:  # e.g. disk error writing status; keep serving later jobs
                traceback.print_exc()

    def _run(self, job_id: str) -> None:
        store = self.store
        status = store.get(job_id)
        if status is None:
            return  # deleted while queued (DELETE /jobs/{id}); nothing left to run
        store.update(job_id, state="training", progress=0.0, message="训练中")

        def on_progress(p: float, message: str | None = None) -> None:
            """message: stage text from the pipeline (precise-mode pose stages); None = by progress."""
            p = min(max(float(p), 0.0), 1.0)
            if message is not None:
                store.update(job_id, state="training", progress=p, message=message)
            elif p >= EXPORT_START:
                store.update(job_id, state="exporting", progress=p, message="导出中")
            else:
                store.update(job_id, state="training", progress=p, message="训练中")

        try:
            note = self.pipeline(store.job_dir(job_id), status.mode, on_progress)
        except Exception as e:
            store.log(job_id, "pipeline failed:\n" + traceback.format_exc().rstrip())
            store.update(job_id, state="failed", message=failure_message(e, job_id))
        else:
            store.update(job_id, state="done", progress=1.0, message=note or "完成")

    def _run_objects(self, job_id: str) -> None:
        store = self.store
        status = store.get(job_id)
        if status is None or status.objects_status.state != "queued":
            return  # deleted, or superseded while queued
        if self.objects_pipeline is None:
            store.update_objects(job_id, state="failed", message="服务端未启用物体识别")
            return
        store.update_objects(job_id, state="running", progress=0.0, message="识别物体")

        def on_progress(p: float, message: str | None = None) -> None:
            p = min(max(float(p), 0.0), 1.0)
            fields = {"progress": p}
            if message is not None:
                fields["message"] = message
            store.update_objects(job_id, state="running", **fields)

        try:
            count = self.objects_pipeline(store.job_dir(job_id), on_progress)
        except Exception as e:
            store.log(job_id, "objects pipeline failed:\n" + traceback.format_exc().rstrip())
            store.update_objects(job_id, state="failed", message=objects_failure_message(e, job_id))
        else:
            store.update_objects(job_id, state="done", progress=1.0, message="完成", count=int(count))
