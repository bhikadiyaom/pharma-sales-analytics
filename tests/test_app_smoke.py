"""Headless smoke test: every dashboard page must render without an exception.

Skipped automatically when the pipeline outputs are not present (run ``make train`` first).
"""

import pytest

from src.config import ROOT, results_dir

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (results_dir() / "headline.json").exists(), reason="pipeline outputs missing; run `make train`"
)

PAGES = ["app/Home.py"] + sorted(
    str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "app/pages").glob("*.py")
)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
