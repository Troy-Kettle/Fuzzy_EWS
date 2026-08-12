"""Root entry point for Streamlit Community Cloud.

The app itself lives in app/streamlit_app.py. The deployment is configured with
``streamlit_app.py`` at the repository root as its main module, so this file loads that
module and runs it — which is why the deployment failed with "The main module file does
not exist" after the app was moved into app/.

Nothing else changes: ``streamlit run app/streamlit_app.py`` still works locally, and so
does .claude/launch.json. Loading by file path (rather than ``import``) avoids the module
being named ``streamlit_app`` twice and shadowing itself.
"""
import importlib.util
from pathlib import Path

_APP_PATH = Path(__file__).resolve().parent / "app" / "streamlit_app.py"

_spec = importlib.util.spec_from_file_location("fuzzy_ews_streamlit_app", _APP_PATH)
if _spec is None or _spec.loader is None:                     # pragma: no cover
    raise ImportError(f"Could not load the Fuzzy EWS app from {_APP_PATH}")
_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

# app/streamlit_app.py only self-starts under `if __name__ == "__main__"`, which does not
# fire when loaded this way, so start it explicitly. main() no-ops outside Streamlit.
_app.main()
