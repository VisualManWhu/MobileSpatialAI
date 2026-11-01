"""HTTP API, see docs/api.md."""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from recon3d import __version__
from recon3d.config import Profile
from recon3d.jobs.store import JobStore, UploadTooLargeError
from recon3d.jobs.worker import Worker

NOT_FOUND = "任务不存在"
NOT_DONE = "任务尚未完成"
NO_GPU = "无 GPU"


@lru_cache(maxsize=1)
def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return NO_GPU


def _too_large(store: JobStore) -> HTTPException:
    gb = store.max_upload_bytes / 1024**3
    return HTTPException(413, f"上传文件过大，上限 {gb:g} GB")


def create_app(store: JobStore, worker: Worker, profile: Profile) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.recover_after_restart()  # before the worker can pick anything up
        worker.start()
        try:
            yield
        finally:
            # Don't hang shutdown on a long training run; the thread is a daemon and the
            # interrupted job is marked failed by recover_after_restart on next start.
            worker.stop(timeout=5.0)

    app = FastAPI(title="Recon3D", version=__version__, lifespan=lifespan)
    api = APIRouter(prefix="/api/v1")

    def status_or_404(job_id: str):
        s = store.get(job_id)
        if s is None:
            raise HTTPException(404, NOT_FOUND)
        return s

    @api.get("/health")
    def health():
        return {"version": __version__, "gpu": gpu_name(), "profile": profile.name}

    @api.post("/jobs", status_code=201)
    async def create_job(request: Request):
        # Reject from the header before the body is received when possible.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > store.max_upload_bytes + 64 * 1024:
            raise _too_large(store)
        # Starlette spools file parts to a temp file (not memory) while parsing.
        form = await request.form(max_files=1, max_fields=8)
        try:
            upload = form.get("capture")
            if not isinstance(upload, UploadFile):
                return JSONResponse({"detail": "缺少 capture 字段（拍摄包 zip）"}, status_code=422)
            try:
                job_id = await run_in_threadpool(store.create, upload.file)
            except UploadTooLargeError:
                raise _too_large(store) from None
        finally:
            await form.close()
        return {"job_id": job_id}

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        s = status_or_404(job_id)
        return {"job_id": s.job_id, "state": s.state, "progress": s.progress, "message": s.message}

    @api.get("/jobs/{job_id}/scene.splat")
    def get_splat(job_id: str):
        s = status_or_404(job_id)
        path = store.job_dir(job_id) / "scene.splat"
        if s.state != "done" or not path.is_file():
            raise HTTPException(409, NOT_DONE)
        return FileResponse(path, media_type="application/octet-stream", filename="scene.splat")

    app.include_router(api)
    return app
