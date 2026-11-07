"""Streaming multipart upload: the `capture` part goes straight into the job's capture.zip.

The body is parsed incrementally from request.stream() (no Starlette form spooling), so the
size cap holds for chunked requests without Content-Length and every byte hits disk once.
"""
from __future__ import annotations

from anyio import to_thread
from fastapi import HTTPException, Request
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from recon3d.jobs.store import MODES, JobStore, Upload, UploadTooLargeError

FIELD = b"capture"
MODE_FIELD = b"mode"
MODE_MAX_LEN = 32
BAD_MODE = "mode 只能是 fast 或 precise"
# Allowance for multipart boundaries, part headers and small extra fields.
FORM_OVERHEAD = 64 * 1024


def too_large(store: JobStore) -> HTTPException:
    gb = store.max_upload_bytes / 1024**3
    return HTTPException(413, f"上传文件过大，上限 {gb:.3g} GB")


class _CaptureSink:
    """MultipartParser callbacks; writes the first `capture` file part to an Upload."""

    def __init__(self, store: JobStore):
        self.store = store
        self.upload: Upload | None = None
        self.complete = False
        self.mode: bytes | None = None  # the optional `mode` text field, in whichever order it came
        self._writing = False
        self._reading_mode = False
        self._field = b""
        self._value = b""
        self._headers: dict[bytes, bytes] = {}

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def _part_begin(self) -> None:
        self._headers = {}
        self._field = self._value = b""

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _header_end(self) -> None:
        self._headers[self._field.lower()] = self._value
        self._field = self._value = b""

    def _headers_finished(self) -> None:
        disp, opts = parse_options_header(self._headers.get(b"content-disposition", b""))
        is_capture = disp == b"form-data" and opts.get(b"name") == FIELD and b"filename" in opts
        if is_capture and self.upload is None:
            self.upload = self.store.begin_upload()
            self._writing = True
        elif disp == b"form-data" and opts.get(b"name") == MODE_FIELD and b"filename" not in opts:
            self.mode = b""
            self._reading_mode = True

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._writing:
            self.upload.write(data[start:end])  # raises UploadTooLargeError past the cap
        elif self._reading_mode and len(self.mode) <= MODE_MAX_LEN:
            self.mode += data[start:end]

    def _part_end(self) -> None:
        if self._writing:
            self._writing = False
            self.complete = True
        self._reading_mode = False


async def receive_capture(request: Request, store: JobStore) -> str:
    """Parse the multipart body, create the job, return its id. Raises HTTPException."""
    mime, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if mime != b"multipart/form-data" or not boundary:
        raise HTTPException(422, "请求必须为 multipart/form-data")

    sink = _CaptureSink(store)
    parser = MultipartParser(boundary, sink.callbacks())
    limit = store.max_upload_bytes + FORM_OVERHEAD
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > limit:  # bounds non-capture parts too
                raise UploadTooLargeError(store.max_upload_bytes)
            if chunk:
                await to_thread.run_sync(parser.write, chunk)
        await to_thread.run_sync(parser.finalize)
        if sink.upload is None:
            raise HTTPException(422, "缺少 capture 字段（拍摄包 zip）")
        if not sink.complete:
            raise HTTPException(400, "上传不完整")
        if sink.mode is not None:
            mode = sink.mode.decode("utf-8", "replace").strip()
            if mode not in MODES:
                raise HTTPException(422, BAD_MODE)
            sink.upload.mode = mode
        upload, sink.upload = sink.upload, None
        return await to_thread.run_sync(upload.commit)
    except UploadTooLargeError:
        raise too_large(store) from None
    except MultipartParseError:
        raise HTTPException(400, "上传格式错误") from None
    finally:
        if sink.upload is not None:  # anything but a successful commit (incl. disconnect)
            sink.upload.abort()
