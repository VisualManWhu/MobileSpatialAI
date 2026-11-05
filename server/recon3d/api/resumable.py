"""Resumable (tus-style) upload endpoints: /api/v1/uploads — see docs/api.md.

The phone registers an upload (size + SHA-256), PUTs byte ranges with an `Upload-Offset` header and
finishes once the server reports every byte received. Bytes are written as they arrive, so a dropped
connection keeps what already landed and the next GET tells the client where to continue.
"""
from __future__ import annotations

import logging

import anyio
from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recon3d.api.upload import too_large
from recon3d.jobs.store import (
    ChecksumMismatch,
    IncompleteUpload,
    JobStore,
    OffsetMismatch,
    UploadBusy,
    UploadTooLargeError,
)

log = logging.getLogger(__name__)

# No body data for this long counts as a dropped connection: a half-open TCP socket must not hold
# the upload's lock for the minutes the kernel takes to notice.
PUT_IDLE_S = 30.0

NOT_FOUND = "上传不存在"
ALREADY_DONE = "上传已完成"
OFFSET_MISMATCH = "偏移不匹配"
BUSY = "该上传正在接收中"
INCOMPLETE = "上传未完成"
CHECKSUM = "文件校验失败，请重新上传"
MISSING_OFFSET = "缺少 Upload-Offset"


class UploadCreate(BaseModel):
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _with_received(status: int, detail: str, received: int) -> JSONResponse:
    return JSONResponse({"detail": detail, "received": received}, status_code=status)


def router(store: JobStore) -> APIRouter:
    api = APIRouter(prefix="/uploads")

    def not_found_or_done(upload_id: str) -> HTTPException:
        if store.get(upload_id) is not None:
            return HTTPException(409, ALREADY_DONE)
        return HTTPException(404, NOT_FOUND)

    @api.post("", status_code=201)
    def create(body: UploadCreate):
        store.sweep_resumable()
        try:
            info = store.begin_resumable(body.size, body.sha256)
        except UploadTooLargeError:
            raise too_large(store) from None
        return {"upload_id": info.upload_id, "received": info.received}

    @api.get("/{upload_id}")
    def info(upload_id: str):
        i = store.resumable(upload_id)
        if i is None:  # finished uploads are 404 too: the client then checks GET /jobs/{id}
            raise HTTPException(404, NOT_FOUND)
        return {"upload_id": i.upload_id, "size": i.size, "received": i.received}

    @api.put("/{upload_id}")
    async def put(upload_id: str, request: Request):
        raw = request.headers.get("upload-offset")
        if raw is None or not raw.isdigit():
            raise HTTPException(422, MISSING_OFFSET)
        offset = int(raw)
        if store.resumable(upload_id) is None:
            raise not_found_or_done(upload_id)

        # The store's writer runs in a worker thread and pulls chunks through a bounded channel the
        # async side fills from request.stream(); the channel closing (disconnect, idle timeout or end
        # of body) ends the writer, which keeps and counts everything written so far.
        send, recv = anyio.create_memory_object_stream[bytes](max_buffer_size=4)

        def chunks():
            with recv:
                while True:
                    try:
                        yield anyio.from_thread.run(recv.receive)
                    except anyio.EndOfStream:
                        return

        async def feed() -> None:
            with send:
                stream = request.stream().__aiter__()
                while True:
                    try:
                        with anyio.fail_after(PUT_IDLE_S):
                            chunk = await stream.__anext__()
                    except (StopAsyncIteration, TimeoutError):
                        return
                    except Exception:  # ClientDisconnect and friends: keep what we have
                        return
                    if chunk:
                        await send.send(chunk)

        outcome: list = []  # [received] or [exception]; a raise inside the task group would be wrapped

        async def write() -> None:
            try:
                outcome.append(await to_thread.run_sync(store.append_resumable, upload_id, offset, chunks()))
            except Exception as e:  # noqa: BLE001 - re-raised below, outside the group
                outcome.append(e)

        async with anyio.create_task_group() as tg:
            tg.start_soon(feed)
            tg.start_soon(write)
        result = outcome[0]
        if isinstance(result, OffsetMismatch):
            return _with_received(409, OFFSET_MISMATCH, result.received)
        if isinstance(result, UploadBusy):
            return _with_received(409, BUSY, result.received)
        if isinstance(result, UploadTooLargeError):
            raise too_large(store)
        if isinstance(result, KeyError):
            raise not_found_or_done(upload_id)
        if isinstance(result, Exception):
            raise result
        return {"received": result}

    @api.post("/{upload_id}/finish", status_code=201)
    def finish(upload_id: str):
        try:
            job_id = store.finish_resumable(upload_id)
        except IncompleteUpload as e:
            return _with_received(409, INCOMPLETE, e.received)
        except ChecksumMismatch:
            raise HTTPException(422, CHECKSUM) from None
        except KeyError:
            raise HTTPException(404, NOT_FOUND) from None
        return {"job_id": job_id}

    return api
