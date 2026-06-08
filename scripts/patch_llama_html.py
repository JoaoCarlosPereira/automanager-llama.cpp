#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "llama_manager.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.strip() == "<script>" and i > 900)
end = next(i for i, l in enumerate(lines) if l.strip() == "</script>" and i > start)
replacement = """    <script>
        window.fixedIp = "{local_ip}";
        window.modelConfigs = window.modelConfigs || {{}};
        window.__constants = {{
            CONTEXT_PRESET_VALUES: {json.dumps(CONTEXT_PRESET_VALUES)},
            DEFAULT_CONTEXT_SIZE: {DEFAULT_CONTEXT_SIZE},
            CONTEXT_K_MULTIPLIER: {CONTEXT_K_MULTIPLIER},
            DEFAULT_PARALLEL_SLOTS: {DEFAULT_PARALLEL_SLOTS},
            DEFAULT_BATCH_SIZE: {DEFAULT_BATCH_SIZE},
        }};
    </script>
    <script type="module" src="/static/js/gpu.js"></script>
    <script type="module" src="/static/js/auth.js"></script>
    <script type="module" src="/static/js/metrics.js"></script>
    <script type="module" src="/static/js/models.js"></script>
    <script type="module" src="/static/js/index.js"></script>
"""
new_lines = lines[:start] + [replacement] + lines[end + 1 :]
path.write_text("".join(new_lines), encoding="utf-8")
print(f"replaced lines {start + 1}-{end + 1}")
