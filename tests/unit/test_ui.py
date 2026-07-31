from __future__ import annotations

import importlib.util
import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_streamlit() -> Generator[MagicMock, None, None]:
    """Mock streamlit modules for testing."""
    with patch("streamlit.set_page_config"), \
         patch("streamlit.markdown"), \
         patch("streamlit.sidebar"), \
         patch("streamlit.text_area"), \
         patch("streamlit.button"), \
         patch("streamlit.toggle"), \
         patch("streamlit.empty"), \
         patch("streamlit.spinner"), \
         patch("streamlit.error"), \
         patch("streamlit.success"), \
         patch("streamlit.warning"), \
         patch("streamlit.divider"), \
         patch("streamlit.expander"), \
         patch("streamlit.file_uploader"):
        
        # Configure st.session_state mock
        st_mock = MagicMock()
        st_mock.session_state = {}
        
        yield st_mock


def test_ui_imports_and_runs_without_exceptions(mock_streamlit: MagicMock) -> None:
    """Verifies that the Streamlit app script can be imported and parsed."""
    _ = mock_streamlit
    
    # Locate ui/app.py path
    ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "app.py")
    ui_path = os.path.abspath(ui_path)
    
    assert os.path.exists(ui_path)
    
    # Load module from path
    spec = importlib.util.spec_from_file_location("streamlit_app", ui_path)
    assert spec is not None
    assert spec.loader is not None
    
    module = importlib.util.module_from_spec(spec)
    
    # We patch httpx calls during load since the app makes a health check on load
    with patch("httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            pytest.fail(f"ui/app.py execution failed: {e}")
