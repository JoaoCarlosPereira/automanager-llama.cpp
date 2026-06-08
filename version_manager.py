"""Git-based version check for self-hosted update notifications."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from log_manager import logger

SHA_SHORT_LEN = 7
LOG_FIELD_SEP = "\x1f"


@dataclass(frozen=True)
class VersionCommit:
    sha: str
    message: str
    author: str
    date: str


@dataclass
class VersionCheckResult:
    status: Literal["ok", "unavailable", "error"]
    update_available: bool = False
    current_ref: Optional[str] = None
    remote_ref: Optional[str] = None
    branch: Optional[str] = None
    commits: List[VersionCommit] = field(default_factory=list)
    error_message: Optional[str] = None


def _short_sha(full_sha: str) -> str:
    value = full_sha.strip()
    return value[:SHA_SHORT_LEN] if len(value) >= SHA_SHORT_LEN else value


def _run_git(
    install_root: str,
    args: List[str],
    *,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", install_root, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_log_output(stdout: str) -> List[VersionCommit]:
    commits: List[VersionCommit] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(LOG_FIELD_SEP, 3)
        if len(parts) != 4:
            logger.warning("Linha de git log ignorada (formato invalido)")
            continue
        sha, message, author, date = parts
        commits.append(
            VersionCommit(
                sha=sha.strip(),
                message=message.strip(),
                author=author.strip(),
                date=date.strip(),
            )
        )
    return commits


def check_for_updates(
    install_root: str,
    fetch_timeout: int = 30,
) -> VersionCheckResult:
    """Compare local HEAD with origin/<current-branch> and list ahead commits."""
    logger.info("Verificacao de versao iniciada em %s", install_root)

    work_tree = _run_git(install_root, ["rev-parse", "--is-inside-work-tree"])
    if work_tree.returncode != 0 or work_tree.stdout.strip() != "true":
        logger.warning("Diretorio nao e um repositorio git: %s", install_root)
        return VersionCheckResult(status="unavailable")

    branch_result = _run_git(install_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch_result.returncode != 0:
        message = (branch_result.stderr or branch_result.stdout or "").strip()
        logger.error("Falha ao obter branch atual: %s", message)
        return VersionCheckResult(status="error", error_message=message or "branch indisponivel")

    branch = branch_result.stdout.strip()
    remote_ref_name = f"origin/{branch}"

    try:
        fetch_result = _run_git(
            install_root,
            ["fetch", "--quiet", "origin", branch],
            timeout=fetch_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("git fetch expirou apos %ss (branch=%s)", fetch_timeout, branch)
        return VersionCheckResult(
            status="error",
            branch=branch,
            error_message=f"git fetch timeout ({fetch_timeout}s)",
        )

    if fetch_result.returncode != 0:
        message = (fetch_result.stderr or fetch_result.stdout or "").strip()
        logger.warning("git fetch falhou (branch=%s): %s", branch, message)
        return VersionCheckResult(
            status="error",
            branch=branch,
            error_message=message or "git fetch falhou",
        )

    local_result = _run_git(install_root, ["rev-parse", "HEAD"])
    remote_result = _run_git(install_root, ["rev-parse", remote_ref_name])
    if local_result.returncode != 0 or remote_result.returncode != 0:
        message = (
            (local_result.stderr or remote_result.stderr or "").strip()
            or "refs locais/remotas indisponiveis"
        )
        logger.error("Falha ao resolver refs: %s", message)
        return VersionCheckResult(
            status="error",
            branch=branch,
            error_message=message,
        )

    local_sha = local_result.stdout.strip()
    remote_sha = remote_result.stdout.strip()
    current_ref = _short_sha(local_sha)
    remote_ref = _short_sha(remote_sha)

    if local_sha == remote_sha:
        logger.info("Versao atualizada (branch=%s, ref=%s)", branch, current_ref)
        return VersionCheckResult(
            status="ok",
            update_available=False,
            current_ref=current_ref,
            remote_ref=remote_ref,
            branch=branch,
        )

    log_result = _run_git(
        install_root,
        [
            "log",
            f"HEAD..{remote_ref_name}",
            f"--format=%H{LOG_FIELD_SEP}%s{LOG_FIELD_SEP}%an{LOG_FIELD_SEP}%aI",
            "--reverse",
        ],
    )
    if log_result.returncode != 0:
        message = (log_result.stderr or log_result.stdout or "").strip()
        logger.error("Falha ao listar commits ahead: %s", message)
        return VersionCheckResult(
            status="error",
            branch=branch,
            current_ref=current_ref,
            remote_ref=remote_ref,
            error_message=message or "git log falhou",
        )

    commits = _parse_log_output(log_result.stdout)
    logger.info(
        "Atualizacao disponivel (branch=%s, ahead=%d, %s -> %s)",
        branch,
        len(commits),
        current_ref,
        remote_ref,
    )
    return VersionCheckResult(
        status="ok",
        update_available=True,
        current_ref=current_ref,
        remote_ref=remote_ref,
        branch=branch,
        commits=commits,
    )
