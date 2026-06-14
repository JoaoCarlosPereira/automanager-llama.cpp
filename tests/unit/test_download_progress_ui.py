"""Contract tests for the download progress UI rendering in metrics.js.

These guard against the regression where the downloads section stopped
displaying the percentage and transfer speed, rendering only the bar.
See updateDownloads() in static/js/metrics.js.
"""

import os

import pytest

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_METRICS_JS = os.path.join(_REPO_ROOT, "static", "js", "metrics.js")


@pytest.fixture(scope="module")
def metrics_js() -> str:
    with open(_METRICS_JS, "r", encoding="utf-8") as fh:
        return fh.read()


def test_has_format_helpers(metrics_js):
    """formatBytes/formatSpeed are required to render size and speed text."""
    assert "function formatBytes(" in metrics_js
    assert "function formatSpeed(" in metrics_js


def test_renders_progress_percentage(metrics_js):
    """Percentage text must be rendered with one decimal place."""
    assert "progress.toFixed(1)" in metrics_js
    assert "%" in metrics_js


def test_renders_download_speed(metrics_js):
    """Transfer speed must be derived from the backend's speed_bps field."""
    assert "formatSpeed(d.speed_bps)" in metrics_js


def test_renders_byte_progress(metrics_js):
    """Downloaded / total bytes must be shown using the backend fields."""
    assert "d.downloaded_bytes" in metrics_js
    assert "d.total_bytes" in metrics_js


def test_renders_download_timing(metrics_js):
    assert "formatDuration" in metrics_js
    assert "d.elapsed_seconds" in metrics_js
    assert "d.eta_seconds" in metrics_js


def test_renders_cancel_download_action(metrics_js):
    assert "cancelDownload" in metrics_js
    assert "/downloads/cancel" in metrics_js


def test_progress_bar_width_uses_progress(metrics_js):
    """The progress bar width must still be bound to the progress value."""
    assert "width: ${progress}%" in metrics_js
