"""Unit tests for version_manager.check_for_updates."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from version_manager import VersionCheckResult, check_for_updates

INSTALL_ROOT = "/tmp/automanager-test"


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _git_key(args):
    return tuple(args[3:])


def test_not_a_git_repo_returns_unavailable():
    with patch("version_manager._run_git") as run_git:
        run_git.return_value = _completed(stdout="false\n")
        result = check_for_updates(INSTALL_ROOT)

    assert result.status == "unavailable"
    assert result.update_available is False


def test_up_to_date_returns_no_update():
    responses = {
        ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
        ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="main\n"),
        ("fetch", "--quiet", "origin", "main"): _completed(),
        ("rev-parse", "HEAD"): _completed(stdout="aaa1111bbb2222\n"),
        ("rev-parse", "origin/main"): _completed(stdout="aaa1111bbb2222\n"),
    }

    def side_effect(install_root, args, timeout=None):
        return responses[tuple(args)]

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT)

    assert result.status == "ok"
    assert result.update_available is False
    assert result.commits == []
    assert result.current_ref == "aaa1111"
    assert result.remote_ref == "aaa1111"
    assert result.branch == "main"


def test_remote_ahead_returns_commits():
    log_line = "ccc3333\x1ffeat: first\x1fAlice\x1f2026-06-01T10:00:00-03:00"
    log_line2 = "ddd4444\x1ffix: second\x1fBob\x1f2026-06-02T11:00:00-03:00"
    responses = {
        ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
        ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="main\n"),
        ("fetch", "--quiet", "origin", "main"): _completed(),
        ("rev-parse", "HEAD"): _completed(stdout="aaa1111bbb2222\n"),
        ("rev-parse", "origin/main"): _completed(stdout="ddd4444eee5555\n"),
        (
            "log",
            "HEAD..origin/main",
            "--format=%H\x1f%s\x1f%an\x1f%aI",
            "--reverse",
        ): _completed(stdout=f"{log_line}\n{log_line2}\n"),
    }

    def side_effect(install_root, args, timeout=None):
        return responses[tuple(args)]

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT)

    assert result.status == "ok"
    assert result.update_available is True
    assert result.current_ref == "aaa1111"
    assert result.remote_ref == "ddd4444"
    assert len(result.commits) == 2
    assert result.commits[0].message == "feat: first"
    assert result.commits[0].author == "Alice"
    assert result.commits[1].message == "fix: second"
    assert result.commits[1].author == "Bob"


def test_fetch_failure_returns_error():
    responses = {
        ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
        ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="main\n"),
        ("fetch", "--quiet", "origin", "main"): _completed(
            stderr="fatal: could not read from remote\n",
            returncode=128,
        ),
    }

    def side_effect(install_root, args, timeout=None):
        return responses.get(tuple(args), _completed())

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT)

    assert result.status == "error"
    assert "remote" in (result.error_message or "").lower()


def test_fetch_timeout_returns_error():
    def side_effect(install_root, args, timeout=None):
        if tuple(args) == ("fetch", "--quiet", "origin", "main"):
            raise subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=30)
        mapping = {
            ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
            ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="main\n"),
        }
        return mapping[tuple(args)]

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT, fetch_timeout=30)

    assert result.status == "error"
    assert "timeout" in (result.error_message or "").lower()


def test_feature_branch_with_slash():
    responses = {
        ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
        ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="feature/foo\n"),
        ("fetch", "--quiet", "origin", "feature/foo"): _completed(),
        ("rev-parse", "HEAD"): _completed(stdout="1111111\n"),
        ("rev-parse", "origin/feature/foo"): _completed(stdout="1111111\n"),
    }

    def side_effect(install_root, args, timeout=None):
        return responses[tuple(args)]

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT)

    assert result.status == "ok"
    assert result.branch == "feature/foo"
    assert result.update_available is False


def test_long_commit_list_not_truncated():
    log_lines = [
        f"sha{i:07d}\x1fmsg {i}\x1fAuthor\x1f2026-06-07T12:00:00-03:00"
        for i in range(12)
    ]
    responses = {
        ("rev-parse", "--is-inside-work-tree"): _completed(stdout="true\n"),
        ("rev-parse", "--abbrev-ref", "HEAD"): _completed(stdout="main\n"),
        ("fetch", "--quiet", "origin", "main"): _completed(),
        ("rev-parse", "HEAD"): _completed(stdout="local0001\n"),
        ("rev-parse", "origin/main"): _completed(stdout="remote001\n"),
        (
            "log",
            "HEAD..origin/main",
            "--format=%H\x1f%s\x1f%an\x1f%aI",
            "--reverse",
        ): _completed(stdout="\n".join(log_lines) + "\n"),
    }

    def side_effect(install_root, args, timeout=None):
        return responses[tuple(args)]

    with patch("version_manager._run_git", side_effect=side_effect):
        result = check_for_updates(INSTALL_ROOT)

    assert result.update_available is True
    assert len(result.commits) == 12
