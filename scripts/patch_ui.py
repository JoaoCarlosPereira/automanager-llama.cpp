"""One-off UI patches for llama_manager.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "llama_manager.py"
text = path.read_text(encoding="utf-8")

old = '                <div class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">'
new = '                <motion id="metrics-panel" class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">'
new = new.replace("motion", "motion")  # noqa: intentional — next line fixes tag
new = '                <div id="metrics-panel" class="grid grid-cols-2 md:grid-cols-2 gap-4 md:gap-6">'
if old not in text:
    raise SystemExit("metrics grid line not found")
text = text.replace(old, new, 1)

chat_old = (
    'id="chat-link" href="#" target="_blank" class="px-6 md:px-10 py-4 bg-blue-600 '
    'hover:bg-blue-500 text-white rounded-2xl'
)
chat_new = (
    'id="chat-link" href="#" target="_blank" class="px-6 md:px-10 py-4 btn-gradient '
    'text-white rounded-2xl pointer-events-none opacity-40"'
)
if chat_old in text:
    text = text.replace(chat_old, chat_new, 1)

online_enable = """                    if (!currentSelectedModel && currentRunningModelPath) {{
                        currentSelectedModel = currentRunningModelPath.replace(/\\\\\\\\/g, '/');
                    }}
                }} else {{"""

online_with_chat = """                    if (!currentSelectedModel && currentRunningModelPath) {{
                        currentSelectedModel = currentRunningModelPath.replace(/\\\\\\\\/g, '/');
                    }}
                    const chatLink = document.getElementById('chat-link');
                    if (chatLink) {{
                        chatLink.classList.remove('pointer-events-none', 'opacity-40');
                        chatLink.setAttribute('aria-disabled', 'false');
                    }}
                }} else {{"""

offline_disable = """                    currentRunningModelPath = null;
                }}"""

offline_with_chat = """                    currentRunningModelPath = null;
                    const chatLinkOff = document.getElementById('chat-link');
                    if (chatLinkOff) {{
                        chatLinkOff.classList.add('pointer-events-none', 'opacity-40');
                        chatLinkOff.setAttribute('aria-disabled', 'true');
                    }}
                }}"""

if online_enable in text and online_with_chat not in text:
    text = text.replace(online_enable, online_with_chat, 1)
if offline_disable in text and offline_with_chat not in text:
    text = text.replace(offline_disable, offline_with_chat, 1)

path.write_text(text, encoding="utf-8")
print("patched", path)
