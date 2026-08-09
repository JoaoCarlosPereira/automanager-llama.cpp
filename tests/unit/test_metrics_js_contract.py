"""Source contracts for periodic dashboard state merging."""

from pathlib import Path


METRICS_JS = (
    Path(__file__).resolve().parents[2] / "static" / "js" / "metrics.js"
).read_text(encoding="utf-8")


def test_polling_preserves_pending_local_model_preferences():
    assert (
        "window.modelConfigs[path] = { ...inst.config, "
        "...(window.modelConfigs[path] || {}) };"
    ) in METRICS_JS
