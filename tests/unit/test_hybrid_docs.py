from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readmes_document_startup_detection_and_credentials_boundary():
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    portuguese = (ROOT / "README.pt-BR.md").read_text(encoding="utf-8")

    assert "Detection runs once when AutoManager starts" in english
    assert "does not collect provider credentials" in english
    assert "primary_backend_id" in english

    assert "A detecção roda uma vez quando o AutoManager inicia" in portuguese
    assert "não coleta credenciais de provedor" in portuguese
    assert "primary_backend_id" in portuguese
