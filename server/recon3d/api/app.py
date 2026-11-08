"""HTTP API, see docs/api.md."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from recon3d import __version__
from recon3d.api.resumable import BUSY
from recon3d.api.resumable import router as resumable_router
from recon3d.api.status_page import HTML as STATUS_PAGE
from recon3d.api.upload import FORM_OVERHEAD, receive_capture
from recon3d.api.upload import too_large as _too_large
from recon3d.config import Profile
from recon3d.jobs.store import JobActive, JobNotDone, JobStore, ObjectsActive, UploadBusy, UploadInProgress
from recon3d.jobs.worker import Worker

log = logging.getLogger(__name__)

NOT_FOUND = "任务不存在"
NOT_DONE = "任务尚未完成"
TRAINING = "任务正在训练，完成后再删"
UPLOAD_INCOMPLETE = "上传未完成"
OBJECTS_ACTIVE = "正在识别"
NO_OBJECTS = "没有物体表"
NO_GPU = "无 GPU"


class RerunBody(BaseModel):
    mode: Literal["fast", "precise"]


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
        """Every job, newest first, for the status page (not used by the phone).

        Uploads still being received are listed first as state "uploading" so the page shows a
        transfer the moment the phone starts it, not only once it becomes a queued job.
        """
        out = []
        for u in store.list_resumable():
            out.append({
                "job_id": u.upload_id,
                "state": "uploading",
                "progress": u.received / u.size,
                "message": f"已收到 {u.received} / {u.size} 字节",
                "created_at": u.created_at,
                "capture_bytes": u.size,
                "has_scene": False,
                "mode": u.mode,
            })
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
                "mode": s.mode,
                "objects": asdict(s.objects_status),
                "capture_id": s.capture_id,
            })
        return {"jobs": out}

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        s = status_or_404(job_id)
        return {
            "job_id": s.job_id,
            "state": s.state,
            "progress": s.progress,
            "message": s.message,
            "mode": s.mode,
            "objects": asdict(s.objects_status),
            "capture_id": s.capture_id,
        }

    @api.post("/jobs/{job_id}/objects", status_code=202)
    def request_objects(job_id: str):
        """Queue object recognition over a done job (phase 2); redoing a finished result is allowed."""
        try:
            store.request_objects(job_id)
        except KeyError:
            raise HTTPException(404, NOT_FOUND) from None
        except JobNotDone:
            raise HTTPException(409, NOT_DONE) from None
        except ObjectsActive:
            raise HTTPException(409, OBJECTS_ACTIVE) from None
        log.info("objects requested for job %s", job_id)
        s = store.get(job_id)
        return {"job_id": job_id, "objects": asdict(s.objects_status)}

    @api.get("/jobs/{job_id}/objects.zip")
    def get_objects(job_id: str):
        s = status_or_404(job_id)
        path = store.job_dir(job_id) / "objects.zip"
        if s.objects_status.state != "done" or not path.is_file():
            raise HTTPException(404, NO_OBJECTS)
        return FileResponse(path, media_type="application/zip", filename="objects.zip")

    @api.post("/jobs/{job_id}/rerun", status_code=201)
    def rerun_job(job_id: str, body: RerunBody):
        """Queue a new job over an existing job's capture.zip in the given mode (hard-linked)."""
        try:
            new_id = store.rerun(job_id, body.mode)
        except KeyError:
            raise HTTPException(404, NOT_FOUND) from None
        except UploadInProgress:
            raise HTTPException(409, UPLOAD_INCOMPLETE) from None
        log.info("rerun job %s -> %s (mode=%s)", job_id, new_id, body.mode)
        return {"job_id": new_id}

    @api.delete("/jobs/{job_id}", status_code=204, response_class=Response)
    def delete_job(job_id: str):
        """Remove a job or an in-progress upload; see docs/api.md for the 404/409 cases."""
        try:
            store.delete(job_id)
        except KeyError:
            raise HTTPException(404, NOT_FOUND) from None
        except JobActive:
            raise HTTPException(409, TRAINING) from None
        except UploadBusy as e:
            return JSONResponse({"detail": BUSY, "received": e.received}, status_code=409)
        log.info("deleted job %s", job_id)
        return Response(status_code=204)

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

    api.include_router(resumable_router(store))
    app.include_router(api)
    return app
