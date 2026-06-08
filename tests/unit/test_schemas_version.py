"""Unit tests for version check response schemas."""

import pytest
from pydantic import ValidationError

from schemas import VersionCheckResponse, VersionCommit


def test_version_check_response_requires_status():
    with pytest.raises(ValidationError):
        VersionCheckResponse()


def test_version_check_response_defaults():
    resp = VersionCheckResponse(status="ok")
    assert resp.update_available is False
    assert resp.commits == []
    assert resp.current_ref is None
    assert resp.remote_ref is None
    assert resp.branch is None
    assert resp.error_message is None


def test_version_check_response_serializes_full_payload():
    resp = VersionCheckResponse(
        status="ok",
        update_available=True,
        current_ref="abc1234",
        remote_ref="def5678",
        branch="main",
        commits=[
            VersionCommit(
                sha="fullsha1",
                message="feat: add version check",
                author="Dev",
                date="2026-06-07T12:00:00-03:00",
            )
        ],
    )
    data = resp.model_dump()
    assert data["status"] == "ok"
    assert data["update_available"] is True
    assert data["current_ref"] == "abc1234"
    assert data["remote_ref"] == "def5678"
    assert data["branch"] == "main"
    assert len(data["commits"]) == 1
    assert data["commits"][0]["message"] == "feat: add version check"


def test_version_commit_accepts_valid_fields():
    commit = VersionCommit(
        sha="abc",
        message="fix: bug",
        author="Author",
        date="2026-01-01T00:00:00Z",
    )
    assert commit.sha == "abc"


@pytest.mark.parametrize("invalid_status", ["pending", "failed", ""])
def test_version_check_response_rejects_invalid_status(invalid_status):
    with pytest.raises(ValidationError):
        VersionCheckResponse(status=invalid_status)
