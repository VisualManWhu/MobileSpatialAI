"""CLI: recon3d-server --profile laptop|server --data-dir PATH --host 0.0.0.0 --port 8765"""
from __future__ import annotations

import argparse
from pathlib import Path

from recon3d.config import PROFILES, Profile
from recon3d.jobs.store import DEFAULT_MAX_UPLOAD_BYTES, JobStore


def build_app(profile: Profile, data_dir: Path, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES):
    from recon3d.api.app import create_app
    from recon3d.jobs.pipeline import run_pipeline
    from recon3d.jobs.worker import Worker

    store = JobStore(data_dir, max_upload_bytes=max_upload_bytes)

    def pipeline(job_dir: Path, mode: str, on_progress) -> str | None:
        return run_pipeline(job_dir, profile, on_progress, mode)

    return create_app(store, Worker(store, pipeline), profile)


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="recon3d-server", description="Recon3D reconstruction server")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-upload-gb", type=float, default=DEFAULT_MAX_UPLOAD_BYTES / 1024**3)
    args = ap.parse_args(argv)

    import logging

    import uvicorn

    # Training progress/pose/depth-loss summaries are logged at INFO by recon3d.recon.*;
    # without this only WARNING and above would reach the console.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    app = build_app(PROFILES[args.profile], args.data_dir, int(args.max_upload_gb * 1024**3))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
