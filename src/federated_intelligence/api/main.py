"""FastAPI application: ``uvicorn federated_intelligence.api.main:app``."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from federated_intelligence import __version__
from federated_intelligence.api.db import make_engine, make_session_factory
from federated_intelligence.api.routes.core import router


def create_app(
    database_url: str | None = None,
    data_dir: str | Path | None = None,
    frontend_dir: str | Path | None = None,
    run_inline: bool = False,
) -> FastAPI:
    app = FastAPI(title="Federated Intelligence API", version=__version__,
                  description="Experiment registry and monitoring API of the federated learning research platform.")
    app.state.engine = make_engine(database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.data_dir = str(data_dir or os.environ.get("FI_DATA_DIR", "data"))
    app.state.run_inline = run_inline  # tests: run experiments synchronously
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(router, prefix="/api")

    fe = Path(frontend_dir or os.environ.get("FI_FRONTEND_DIR", "frontend/dist"))
    if (fe / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=fe / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):  # client-side routing fallback
            f = fe / path
            return FileResponse(f if path and f.is_file() else fe / "index.html")
    return app


app = create_app()
