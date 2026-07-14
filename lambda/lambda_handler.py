"""
Lambda entrypoint - wraps the existing FastAPI app (unchanged) with Mangum,
the standard adapter that translates Lambda's event/context format into
ASGI calls the FastAPI app already understands. The app's actual logic
(/predict, /health) doesn't need to know it's running in Lambda at all -
this is the only Lambda-specific file in the whole project.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# Local repo layout: this file lives in lambda/, with app/ as a sibling one
# level up. Lambda container layout (see lambda/Dockerfile): this file is
# copied directly to the task root, with app/ as its own direct sibling -
# one level up would escape the task root entirely. Handle both without
# needing two versions of this file, and without ever importing a
# directory literally named "lambda" - that's a reserved Python keyword
# and can never be a valid package name.
_candidates = [_here.parent / "app", _here / "app"]
_app_dir = next((p for p in _candidates if (p / "main.py").exists()), None)
if _app_dir is None:
    raise ImportError(f"Could not locate app/main.py near {_here}")
sys.path.insert(0, str(_app_dir))

from main import app  # noqa: E402 - must follow sys.path fix above, not a real issue
from mangum import Mangum  # noqa: E402 - same reason

handler = Mangum(app)

