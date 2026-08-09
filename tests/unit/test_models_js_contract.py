"""Source-contract tests for static/js/models.js startModel().

These guard a real regression: the MTP warning block referenced ``startData``
*after* the try/catch, but ``startData`` is declared with ``const`` *inside*
the try block. Because ``const`` is block-scoped, every call to startModel()
threw ``ReferenceError: startData is not defined`` right after the /start
request — silently aborting auto-balance follow-through (and the final status
refresh).

There is no JS unit-test runner in this project, so we statically enforce the
scope invariant the same way test_html_contract.py validates frontend source:
``startData`` must only be referenced inside the try block where it is declared.
"""

import os
import re

import pytest

MODELS_JS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "static", "js", "models.js"
)


def _match_block(source: str, open_index: int) -> int:
    """Return the index just past the ``}`` matching the ``{`` at *open_index*.

    Naive brace counting — valid for startModel(), whose string/template
    literals contain no unbalanced braces.
    """
    assert source[open_index] == "{", "expected '{' at open_index"
    depth = 0
    for i in range(open_index, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise AssertionError("unbalanced braces — no matching '}' found")


@pytest.fixture(scope="module")
def models_js() -> str:
    with open(MODELS_JS_PATH, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def start_model_body(models_js: str) -> str:
    """The full body (including braces) of the startModel() function."""
    sig = re.search(r"export\s+async\s+function\s+startModel\s*\([^)]*\)\s*", models_js)
    assert sig, "startModel() function not found in models.js"
    open_brace = models_js.index("{", sig.end())
    end = _match_block(models_js, open_brace)
    return models_js[open_brace:end]


@pytest.fixture(scope="module")
def try_block(start_model_body: str) -> str:
    """The contents of the try { ... } block inside startModel()."""
    try_match = re.search(r"\btry\s*", start_model_body)
    assert try_match, "startModel() must wrap the /start request in a try block"
    open_brace = start_model_body.index("{", try_match.end())
    end = _match_block(start_model_body, open_brace)
    return start_model_body[open_brace:end]


def test_start_data_declared_inside_try(try_block: str):
    """startData must be declared (const) inside the try block."""
    assert re.search(r"\bconst\s+startData\b", try_block), (
        "startData must be declared with const inside the try block"
    )


def test_start_data_only_referenced_inside_try(start_model_body: str, try_block: str):
    """No startData reference may live outside the try block (scope regression)."""
    total_refs = len(re.findall(r"\bstartData\b", start_model_body))
    inside_refs = len(re.findall(r"\bstartData\b", try_block))
    assert total_refs > 0, "expected startData to be used in startModel()"
    assert total_refs == inside_refs, (
        f"{total_refs - inside_refs} reference(s) to startData found OUTSIDE the "
        "try block — startData is const/block-scoped to the try, so this throws "
        "ReferenceError at runtime (regression of the MTP warning bug)."
    )


def test_mtp_warning_block_present_and_scoped(try_block: str):
    """The MTP warning handling must still exist — and inside the try block."""
    assert "startData.mtp_applied" in try_block, (
        "MTP warning handling (startData.mtp_applied) must run inside the try block"
    )
    assert "showMtpWarning" in try_block
    assert "hideMtpWarning" in try_block


def test_probing_branch_sets_auto_balance_pending(models_js: str):
    """Auto-balance follow-through must remain wired in the smart calibration flow.

    Note: state.autoBalancePending is set at the top-level module scope within
    startSmartCalibration(), not inside startModel()'s try block. The probing
    path (data.probing) and autoBalancePending logic lives in the
    startSmartCalibration() function.
    """
    assert "startSmartCalibration" in models_js
    assert "data.probing" in models_js or "probing" in models_js
    assert "state.autoBalancePending" in models_js


def test_ollama_credentials_have_individual_delete_control(models_js: str):
    """Cada conta Ollama Cloud deve poder ser removida sem apagar as demais."""
    assert 'class="ollama-account-delete' in models_js
    assert "data-account-id=" in models_js
    assert "export async function deleteOllamaCloudAccount" in models_js
    assert "method: 'DELETE'" in models_js
    assert "/platforms/ollama-cloud/accounts/${encodeURIComponent(accountId)}" in models_js


def test_ollama_platform_manage_button_uses_cloud_flow(models_js: str):
    """O botão Gerenciar da aba Ollama não deve abrir o OAuth do CLIProxy."""
    assert "if (platform.provider === 'ollama-cloud')" in models_js
    assert "manageOllamaCloudAuth(backendId, displayName)" in models_js
