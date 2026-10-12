"""WriteTrace API entry point for Vercel (Phase 6B; frontend routing 6D.1).

Vercel's Python runtime looks for an entrypoint file at the project root
(`app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`, `asgi.py`) and loads
the top-level variable named `app`
(https://vercel.com/docs/functions/runtimes/python).

This file does nothing except re-export the ONE authoritative FastAPI
application defined in `backend/app.py`, and then point FastAPI at the built
Astro site so the static frontend and the API share ONE origin:

    POST /api/v1/analyze    -> backend.schemas.AnalyzeRequest/Response
    GET  /api/v1/health     -> liveness
    GET  /api/v1/info       -> model/gate/label metadata
    GET  /                  -> the built Astro site (frontend/dist)

Because both halves share an origin the browser calls /api/v1/analyze
same-origin and no CORS configuration is required in production.

Routing is handled by FastAPI itself (https://fastapi.tiangolo.com/tutorial/frontend/):
"FastAPI checks path operations first. The frontend files are checked only if
no normal route matched, so your API won't be affected." The API contract
therefore keeps precedence over frontend files regardless of Vercel.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The bundled function's working directory is the project root; make the
# repository root importable so `ml.models.final-v1` resolves relative to it.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402,F401  (re-exported for Vercel)

# --- serve the built Astro site from this same origin (Phase 6D.1) --------
# `npm run build` (the vercel.json buildCommand) writes the static site to
# frontend/dist/. FastAPI serves those files; unmatched paths fall through to
# the normal 404 rather than an index.html fallback, because WriteTrace is a
# single-page site with no client-side router and generates no 404.html.
#
# check_dir=False is the documented setting for a separately-built frontend
# ("If your frontend files are created later, for example by a separate build
# step after the app object is created"). Without it FastAPI would raise at
# import time whenever the build step has not yet produced the directory.
app.frontend("/", directory="frontend/dist", fallback=None, check_dir=False)

__all__ = ["app"]