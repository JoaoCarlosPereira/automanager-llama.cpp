#!/usr/bin/env python3
"""Pick an official llama.cpp release asset for the AutoManager installer.

The runtime consumes the ``llama-server`` binary (and ``llama-gguf`` beside it),
not a source tree. Asset names follow the GitHub releases documented in
``installer/setup.sh`` and ``.docs/tasks/vision-por-modelo/adrs/adr-001.md``
(https://github.com/ggml-org/llama.cpp).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Iterable, Optional


LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.split("."):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    return tuple(parts) or (0,)


def select_llama_cpp_assets(
    releases: Iterable[dict[str, Any]],
    *,
    arch: str,
    cuda_version: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Return the newest usable Ubuntu asset for ``arch`` (``x64`` or ``arm64``).

    CUDA builds are preferred. When the driver CUDA version is known, the
    highest published CUDA build that the driver can load is chosen. A newer
    CUDA build than the driver is skipped. When the driver version is unknown,
    the oldest CUDA build in that release is chosen. If no CUDA asset fits,
    the CPU Ubuntu archive for the same tag is used.
    """
    if arch not in ("x64", "arm64"):
        raise ValueError(f"unsupported arch: {arch}")

    cuda_re = re.compile(
        rf"^llama-(b\d+)-bin-ubuntu-cuda-(\d+\.\d+)-{re.escape(arch)}\.tar\.gz$"
    )
    cpu_re = re.compile(rf"^llama-(b\d+)-bin-ubuntu-{re.escape(arch)}\.tar\.gz$")
    cudart_re = re.compile(
        rf"^cudart-llama-(b\d+)-bin-ubuntu-cuda-(\d+\.\d+)-{re.escape(arch)}\.tar\.gz$"
    )
    driver = _version_tuple(cuda_version) if cuda_version else None

    for release in releases:
        if not isinstance(release, dict):
            continue
        tag = str(release.get("tag_name") or "")
        if not tag:
            continue
        names = [
            asset.get("name")
            for asset in release.get("assets") or []
            if isinstance(asset, dict) and isinstance(asset.get("name"), str)
        ]
        cuda_assets: list[tuple[str, tuple[int, ...], str]] = []
        for name in names:
            match = cuda_re.match(name)
            if match and match.group(1) == tag:
                cuda_assets.append((name, _version_tuple(match.group(2)), match.group(2)))

        chosen = ""
        variant = ""
        if cuda_assets:
            if driver is not None:
                eligible = [item for item in cuda_assets if item[1] <= driver]
                if eligible:
                    name, _, label = max(eligible, key=lambda item: item[1])
                    chosen = name
                    variant = f"cuda-{label}"
            else:
                name, _, label = min(cuda_assets, key=lambda item: item[1])
                chosen = name
                variant = f"cuda-{label}"

        if not chosen:
            for name in names:
                match = cpu_re.match(name)
                if match and match.group(1) == tag:
                    chosen = name
                    variant = "cpu"
                    break
        if not chosen:
            continue

        cudart = ""
        if variant.startswith("cuda-"):
            cuda_label = variant.split("-", 1)[1]
            for name in names:
                match = cudart_re.match(name)
                if match and match.group(1) == tag and match.group(2) == cuda_label:
                    cudart = name
                    break

        return {
            "tag": tag,
            "asset": chosen,
            "cudart": cudart,
            "variant": variant,
            "repo": LLAMA_CPP_REPO,
        }
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("x64", "arm64"))
    parser.add_argument("--cuda", default="")
    args = parser.parse_args(argv)
    try:
        releases = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"invalid release JSON: {exc}", file=sys.stderr)
        return 1
    if isinstance(releases, dict):
        releases = [releases]
    if not isinstance(releases, list):
        print("release JSON must be a list", file=sys.stderr)
        return 1
    selected = select_llama_cpp_assets(
        releases,
        arch=args.arch,
        cuda_version=args.cuda.strip() or None,
    )
    if not selected:
        print("no matching llama.cpp release asset", file=sys.stderr)
        return 1
    print(
        "\t".join(
            [
                selected["tag"],
                selected["asset"],
                selected["cudart"],
                selected["variant"],
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
