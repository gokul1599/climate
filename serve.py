"""Root-level server entrypoint for MonsoonVerse (python serve.py --port 8000)."""
from __future__ import annotations

import argparse
import uvicorn

from backend.api import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MonsoonVerse Inference Server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
