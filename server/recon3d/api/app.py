"""HTTP API, see docs/api.md."""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from recon3d import __version__
from recon3d.api.status_page import HTML as STATUS_PAGE
from recon3d.api.upload import FORM_OVERHEAD, receive_capture
from recon3d.api.upload import too_large as _too_large
from recon3d.config import Profile
from recon3d.jobs.store import JobStore
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
        # Reject from the header before reading the body when possible.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > store.max_upload_bytes + FORM_OVERHEAD:
            raise _too_large(store)
        job_id = await receive_capture(request, store)
        return {"job_id": job_id}

    @api.get("/jobs")
    def list_jobs():
        """Every job, newest first, for the status page (not used by the phone)."""
        out = []
        for s in store.list():
            d = store.job_dir(s.job_id)
            try:
                size = (d / "capture.zip").stat().st_size
            except OSError:
                size = 0
            out.append({
                "job_id": s.job_id,
                "state": s.state,
                "progress": s.progress,
                "message": s.message,
                "created_at": s.created_at,
                "capture_bytes": size,
                "has_scene": s.state == "done" and (d / "scene.splat").is_file(),
            })
        return {"jobs": out}

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

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def status_page():
        return STATUS_PAGE

    app.include_router(api)
    return app
