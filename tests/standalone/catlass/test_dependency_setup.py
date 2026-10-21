# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_NAME = "catlass-41bf90da-headers.tar.gz"


@pytest.fixture
def source_archive(tmp_path):
    root = tmp_path / "source with spaces"
    root.mkdir()
    for name in (".gitmodules", ARCHIVE_NAME, ARCHIVE_NAME + ".sha256"):
        shutil.copy2(REPO_ROOT / name, root / name)
    return root


def run_setup(root, tmp_path, cpath=""):
    source = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    match = re.search(r"(?ms)^setup_catlass_dependency\(\) \{.*?^\}", source)
    assert match is not None
    harness = tmp_path / "setup.sh"
    harness.write_text(
        'set -eu\nROOT_DIR="$1"\nlog() { echo "[test] $*"; }\n'
        + match.group()
        + '\nsetup_catlass_dependency\nprintf "CPATH=%s\\n" "$CPATH"\n'
    )
    env = os.environ.copy()
    env.update(
        CPATH=cpath,
        GIT_CONFIG_GLOBAL=str(tmp_path / "global-git-config"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_ALLOW_PROTOCOL="file",
        GIT_TERMINAL_PROMPT="0",
    )
    return subprocess.run(
        ["bash", str(harness), str(root)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_restores_bundled_headers_without_git_metadata(source_archive, tmp_path):
    result = run_setup(source_archive, tmp_path, cpath="/existing/includes")
    assert result.returncode == 0, result.stdout + result.stderr
    include = source_archive / "csrc/third_party/catlass/include"
    assert (include / "catlass/catlass.hpp").is_file()
    assert (include / "tla/layout.hpp").is_file()
    assert f"CPATH={include}:/existing/includes" in result.stdout
    assert "restoring CATLASS" in result.stdout
    assert not (tmp_path / "global-git-config").exists()

    (source_archive / ARCHIVE_NAME).unlink()
    repeat = run_setup(source_archive, tmp_path)
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    assert "restoring CATLASS" not in repeat.stdout


def test_rejects_corrupt_archive_before_extracting(source_archive, tmp_path):
    with (source_archive / ARCHIVE_NAME).open("ab") as stream:
        stream.write(b"corrupted during transfer")
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "checksum verification failed" in result.stdout
    assert not (source_archive / "csrc/third_party/catlass/include").exists()


def test_requires_checksum_file(source_archive, tmp_path):
    (source_archive / (ARCHIVE_NAME + ".sha256")).unlink()
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "checksum verification failed" in result.stdout
    assert not (source_archive / "csrc/third_party/catlass/include").exists()


def test_reports_extraction_failure(source_archive, tmp_path):
    archive = source_archive / ARCHIVE_NAME
    archive.write_bytes(b"not a tar archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (source_archive / (ARCHIVE_NAME + ".sha256")).write_text(f"{digest}  {ARCHIVE_NAME}\n")
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "archive extraction failed" in result.stdout


def test_explains_missing_dependency_in_source_archive(source_archive, tmp_path):
    (source_archive / ARCHIVE_NAME).unlink()
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "no Git metadata" in result.stdout
    assert ARCHIVE_NAME in result.stdout
    assert "filesystem boundary" not in result.stderr


@pytest.mark.parametrize("commit", [None, "master"])
def test_requires_explicit_pinned_commit(source_archive, tmp_path, commit):
    if commit is None:
        (source_archive / ".gitmodules").unlink()
    else:
        (source_archive / ".gitmodules").write_text(f'[submodule "csrc/third_party/catlass"]\ncommit = {commit}\n')
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "pinned CATLASS commit" in result.stdout


def test_rejects_incomplete_existing_headers(source_archive, tmp_path):
    (source_archive / "csrc/third_party/catlass/include").mkdir(parents=True)
    result = run_setup(source_archive, tmp_path)
    assert result.returncode != 0
    assert "headers are incomplete" in result.stdout


def test_git_checkout_can_fetch_pinned_submodule_without_bundle(tmp_path):
    # Use a real local Git remote so the fallback is tested without Internet access.
    def git(root, *args):
        env = os.environ.copy()
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_ALLOW_PROTOCOL="file")
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return result.stdout.strip()

    def init(root):
        root.mkdir()
        git(root, "init")
        git(root, "config", "user.name", "Dependency test")
        git(root, "config", "user.email", "test@example.invalid")
        git(root, "config", "commit.gpgsign", "false")

    upstream = tmp_path / "upstream"
    init(upstream)
    for name in ("include/catlass/catlass.hpp", "include/tla/layout.hpp"):
        header = upstream / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("// pinned test header\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "Add headers")
    commit = git(upstream, "rev-parse", "HEAD")

    origin = tmp_path / "origin"
    init(origin)
    git(origin, "submodule", "add", str(upstream), "csrc/third_party/catlass")
    git(origin, "config", "-f", ".gitmodules", "submodule.csrc/third_party/catlass.commit", commit)
    git(origin, "add", ".")
    git(origin, "commit", "-m", "Pin dependency")
    root = tmp_path / "checkout"
    git(tmp_path, "clone", str(origin), str(root))

    result = run_setup(root, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    dependency = root / "csrc/third_party/catlass"
    assert (dependency / "include/catlass/catlass.hpp").read_text() == "// pinned test header\n"
    assert git(dependency, "rev-parse", "HEAD") == commit
    assert "try to fetch the submodule" in result.stdout
    assert not (tmp_path / "global-git-config").exists()
