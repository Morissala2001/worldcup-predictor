"""Smoke test: the Streamlit app renders every tab without raising."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_exception():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Single match", "Tournament", "Model"]
    assert "Predicted winner" in app.subheader[0].value


def test_app_rejects_identical_teams():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox[1].select(app.selectbox[0].value).run()
    assert not app.exception
    assert any("two different teams" in w.value for w in app.warning)
