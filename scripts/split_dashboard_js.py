#!/usr/bin/env python3
"""Split extracted dashboard JS into ES modules (auth, gpu, metrics, models, index)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "static" / "js" / "_extracted_raw.js"
OUT = ROOT / "static" / "js"

CONST_REPLACEMENTS = [
    ("const fixedIp = ", "// fixedIp set in HTML\n"),
    ("const CONTEXT_PRESET_VALUES = ", "// CONTEXT_PRESET_VALUES in window.__constants\n"),
    ("const DEFAULT_CONTEXT_SIZE_UI = ", "// DEFAULT_CONTEXT_SIZE_UI in window.__constants\n"),
    ("const CONTEXT_K_MULTIPLIER = ", "// CONTEXT_K_MULTIPLIER in window.__constants\n"),
    ("DEFAULT_CONTEXT_SIZE_UI", "window.__constants.DEFAULT_CONTEXT_SIZE"),
    ("CONTEXT_K_MULTIPLIER", "window.__constants.CONTEXT_K_MULTIPLIER"),
    ("CONTEXT_PRESET_VALUES", "window.__constants.CONTEXT_PRESET_VALUES"),
    ('document.getElementById(\'parallel-slots\').value = "{DEFAULT_PARALLEL_SLOTS}";',
     "document.getElementById('parallel-slots').value = String(window.__constants.DEFAULT_PARALLEL_SLOTS);"),
    ('document.getElementById(\'batch-size\').value = "{DEFAULT_BATCH_SIZE}";',
     "document.getElementById('batch-size').value = String(window.__constants.DEFAULT_BATCH_SIZE);"),
    ("|| {DEFAULT_PARALLEL_SLOTS}", "|| window.__constants.DEFAULT_PARALLEL_SLOTS"),
    ("|| {DEFAULT_BATCH_SIZE}", "|| window.__constants.DEFAULT_BATCH_SIZE"),
    ("`${fixedIp}", "`${window.fixedIp}"),
    ("http://${fixedIp}", "http://${window.fixedIp}"),
]

GPU_FUNCS = {
    "syncContextSizeCustomVisibility", "tokensToContextK", "onContextSizePresetChange",
    "onContextSizeCustomInput", "getContextSize", "setContextSize", "markManualGpuChange",
    "applyGpuWeightsToUI", "cancelAutoBalance", "hideAutoBalanceCapacityAlert",
    "showAutoBalanceCapacityAlert", "modelIncapableBadgeHtml", "modelIncapableRowClass",
    "isModelHardwareIncapable", "updateAutoBalanceProfileBadge", "syncAutoBalanceCancelButton",
    "bindGpuManualListeners", "onGpuPinToggle", "redistributeUnpinnedWeights",
    "balanceWeights", "updateTotal", "resetToDefaults",
}

AUTH_FUNCS = {
    "handleSessionExpired", "apiFetch", "handleLogin", "handleLogout", "changePassword",
}

METRICS_FUNCS = {
    "stopDashboardPolling", "startDashboardPolling", "ensureStatusPolling",
    "updateMetrics", "startLogs", "updateUptime", "renewToken", "updateStatus",
    "updateDownloads",
}

MODELS_FUNCS = {
    "initDashboard", "getModelButtonsHtml", "selectModel", "applyModelConfig",
    "setDefaultModel", "downloadModel", "updateModels", "renameModel", "deleteModel",
    "startModel", "stopModel",
}

STATE_VARS = [
    "logStream", "startTime", "currentSelectedModel", "currentRunningModelPath",
    "manualGpuOverride", "autoBalancePending", "sessionExpiredHandled",
    "metricsTimer", "downloadsTimer", "modelsTimer", "statusPollIntervalMs", "statusPollTimer",
]


def transform_body(text: str) -> str:
    for old, new in CONST_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _brace_delta(line: str) -> int:
    """Count { and } outside strings/template literals (approximate)."""
    delta = 0
    i = 0
    n = len(line)
    state = "code"  # code | sq | dq | tpl
    while i < n:
        ch = line[i]
        if state == "code":
            if ch == "'":
                state = "sq"
            elif ch == '"':
                state = "dq"
            elif ch == "`":
                state = "tpl"
            elif ch == "{":
                delta += 1
            elif ch == "}":
                delta -= 1
        elif state == "sq":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "code"
        elif state == "dq":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "code"
        elif state == "tpl":
            if ch == "\\":
                i += 1
            elif ch == "`":
                state = "code"
            elif ch == "$" and i + 1 < n and line[i + 1] == "{":
                i += 1
                depth = 1
                i += 1
                while i < n and depth:
                    c2 = line[i]
                    if c2 == "{":
                        depth += 1
                    elif c2 == "}":
                        depth -= 1
                    i += 1
                continue
        i += 1
    return delta


def parse_functions(source: str) -> dict[str, str]:
    lines = source.splitlines(keepends=True)
    funcs = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("let ") or stripped.startswith("const ") or stripped.startswith("//"):
            i += 1
            continue
        if stripped.startswith("document.getElementById('chat-link')"):
            i += 2
            continue
        if not (stripped.startswith("async function ") or stripped.startswith("function ")):
            i += 1
            continue
        if stripped.startswith("async function "):
            name = stripped.split("async function ", 1)[1].split("(")[0].strip()
        else:
            name = stripped.split("function ", 1)[1].split("(")[0].strip()
        start = i
        depth = _brace_delta(lines[i])
        started = "{" in lines[i]
        i += 1
        while i < len(lines):
            depth += _brace_delta(lines[i])
            if "{" in lines[i]:
                started = True
            if started and depth == 0:
                i += 1
                break
            i += 1
        funcs[name] = "".join(lines[start:i])
    return funcs


def module_header(imports: list[str]) -> str:
    parts = [imports[0], ""] if imports else [""]
    if len(imports) > 1:
        parts = imports + [""]
    return "\n".join(parts)


def emit_gpu(funcs: dict) -> str:
    imports = [
        "import * as state from './state.js';",
        "import { apiFetch } from './auth.js';",
    ]
    lines = imports + [""]
    for name in sorted(GPU_FUNCS, key=lambda n: list(GPU_FUNCS).index(n) if n in GPU_FUNCS else 0):
        if name not in funcs:
            continue
        body = transform_body(funcs[name])
        if "async function " in body.split("\n", 1)[0]:
            body = body.replace("async function ", "export async function ", 1)
        else:
            body = body.replace("function ", "export function ", 1)
        for var in STATE_VARS:
            body = body.replace(f"{var}", f"state.{var}")
        lines.append(body)
    return "\n".join(lines)


def emit_auth(funcs: dict) -> str:
    lines = [
        "import * as state from './state.js';",
        "",
    ]
    for name in ["handleSessionExpired", "apiFetch", "handleLogin", "handleLogout", "changePassword"]:
        body = transform_body(funcs[name])
        if "async function " in body.split("\n", 1)[0]:
            body = body.replace("async function ", "export async function ", 1)
        else:
            body = body.replace("function ", "export function ", 1)
        for var in STATE_VARS:
            body = body.replace(f"{var}", f"state.{var}")
        body = body.replace("stopDashboardPolling()", "window.stopDashboardPolling()")
        body = body.replace("initDashboard()", "window.initDashboard()")
        body = body.replace("startDashboardPolling()", "window.startDashboardPolling()")
        lines.append(body)
    lines += [
        "export function showAlert(msg) { alert(msg); }",
        "export function showConfirm(msg) { return confirm(msg); }",
        "",
    ]
    return "\n".join(lines)


def emit_metrics(funcs: dict) -> str:
    lines = [
        "import * as state from './state.js';",
        "import { apiFetch } from './auth.js';",
        "import {",
        "    applyGpuWeightsToUI, getContextSize, setContextSize,",
        "    hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,",
        "    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,",
        "} from './gpu.js';",
        "",
    ]
    order = list(METRICS_FUNCS)
    for name in order:
        if name not in funcs:
            continue
        body = transform_body(funcs[name])
        if "async function " in body.split("\n", 1)[0]:
            body = body.replace("async function ", "export async function ", 1)
        else:
            body = body.replace("function ", "export function ", 1)
        for var in STATE_VARS:
            body = body.replace(f"{var}", f"state.{var}")
        body = body.replace("updateModels()", "window.updateModels()")
        body = body.replace("getModelButtonsHtml(", "window.getModelButtonsHtml(")
        lines.append(body)
    return "\n".join(lines)


def emit_models(funcs: dict) -> str:
    lines = [
        "import * as state from './state.js';",
        "import { apiFetch } from './auth.js';",
        "import {",
        "    getContextSize, setContextSize, resetToDefaults, applyGpuWeightsToUI,",
        "    updateTotal, hideAutoBalanceCapacityAlert, showAutoBalanceCapacityAlert,",
        "    updateAutoBalanceProfileBadge, syncAutoBalanceCancelButton,",
        "    isModelHardwareIncapable, modelIncapableBadgeHtml, modelIncapableRowClass,",
        "    bindGpuManualListeners, syncContextSizeCustomVisibility,",
        "} from './gpu.js';",
        "import { startLogs, updateUptime } from './metrics.js';",
        "",
    ]
    init = transform_body(funcs["initDashboard"])
    if "async function " in init.split("\n", 1)[0]:
        init = init.replace("async function ", "export async function ", 1)
    else:
        init = init.replace("function ", "export function ", 1)
    init = init.replace("updateStatus()", "window.updateStatus()")
    init = init.replace("updateMetrics()", "window.updateMetrics()")
    init = init.replace("updateDownloads()", "window.updateDownloads()")
    init = init.replace("updateModels()", "window.updateModels()")
    lines.append(init)
    for name in [
        "getModelButtonsHtml", "selectModel", "applyModelConfig", "setDefaultModel",
        "downloadModel", "updateModels", "renameModel", "deleteModel", "startModel", "stopModel",
    ]:
        if name not in funcs:
            continue
        body = transform_body(funcs[name])
        if "async function " in body.split("\n", 1)[0]:
            body = body.replace("async function ", "export async function ", 1)
        else:
            body = body.replace("function ", "export function ", 1)
        for var in STATE_VARS:
            body = body.replace(f"{var}", f"state.{var}")
        body = body.replace("updateStatus", "window.updateStatus")
        body = body.replace("updateMetrics", "window.updateMetrics")
        body = body.replace("updateDownloads", "window.updateDownloads")
        body = body.replace("ensureStatusPolling", "window.ensureStatusPolling")
        lines.append(body)
    return "\n".join(lines)


def emit_state() -> str:
    return """// Shared dashboard state
export let logStream = null;
export let startTime = null;
export let currentSelectedModel = null;
export let currentRunningModelPath = null;
export let manualGpuOverride = false;
export let autoBalancePending = false;
export let sessionExpiredHandled = false;
export let metricsTimer = null;
export let downloadsTimer = null;
export let modelsTimer = null;
export let statusPollIntervalMs = 3000;
export let statusPollTimer = null;
"""


def emit_index() -> str:
    return """import * as state from './state.js';
import { handleLogin, handleLogout, changePassword, apiFetch } from './auth.js';
import { cancelAutoBalance } from './gpu.js';
import {
    syncContextSizeCustomVisibility, onContextSizePresetChange, onContextSizeCustomInput,
    getContextSize, setContextSize, balanceWeights, updateTotal, resetToDefaults,
    hideAutoBalanceCapacityAlert, } from './gpu.js';
import {
    stopDashboardPolling, startDashboardPolling, renewToken, updateMetrics, updateStatus,
} from './metrics.js';
import {
    initDashboard, getModelButtonsHtml, selectModel, applyModelConfig, setDefaultModel,
    startModel, stopModel, renameModel, deleteModel, downloadModel, updateModels,
} from './models.js';

window.modelConfigs = window.modelConfigs || {};

const win = window;
win.initDashboard = initDashboard;
win.startDashboardPolling = startDashboardPolling;
win.stopDashboardPolling = stopDashboardPolling;
win.handleLogin = handleLogin;
win.handleLogout = handleLogout;
win.changePassword = changePassword;
win.apiFetch = apiFetch;
win.cancelAutoBalance = cancelAutoBalance;
win.onContextSizePresetChange = onContextSizePresetChange;
win.onContextSizeCustomInput = onContextSizeCustomInput;
win.getContextSize = getContextSize;
win.setContextSize = setContextSize;
win.stopModel = stopModel;
win.startModel = startModel;
win.renameModel = renameModel;
win.deleteModel = deleteModel;
win.setDefaultModel = setDefaultModel;
win.selectModel = selectModel;
win.applyModelConfig = applyModelConfig;
win.renewToken = renewToken;
win.updateTotal = updateTotal;
win.balanceWeights = balanceWeights;
win.resetToDefaults = resetToDefaults;
win.hideAutoBalanceCapacityAlert = hideAutoBalanceCapacityAlert;
win.getModelButtonsHtml = getModelButtonsHtml;
win.updateModels = updateModels;
win.updateStatus = updateStatus;
win.downloadModel = downloadModel;
win.updateMetrics = updateMetrics;

document.getElementById('chat-link').href = `http://${window.fixedIp}:8085/`;
document.getElementById('api-link').innerText = `http://${window.fixedIp}:8085/v1`;

if (document.getElementById('dashboard').style.display !== 'none') {
    initDashboard();
    startDashboardPolling();
}
"""


def main():
    source = RAW.read_text(encoding="utf-8")
    funcs = parse_functions(source)
    missing = (
        GPU_FUNCS | AUTH_FUNCS | METRICS_FUNCS | MODELS_FUNCS
    ) - set(funcs)
    if missing:
        raise SystemExit(f"Missing functions: {sorted(missing)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state.js").write_text(emit_state(), encoding="utf-8")
    (OUT / "gpu.js").write_text(emit_gpu(funcs), encoding="utf-8")
    (OUT / "auth.js").write_text(emit_auth(funcs), encoding="utf-8")
    (OUT / "metrics.js").write_text(emit_metrics(funcs), encoding="utf-8")
    (OUT / "models.js").write_text(emit_models(funcs), encoding="utf-8")
    (OUT / "index.js").write_text(emit_index(), encoding="utf-8")
    print("Wrote modules:", sorted(f.name for f in OUT.glob("*.js") if not f.name.startswith("_")))


if __name__ == "__main__":
    main()
