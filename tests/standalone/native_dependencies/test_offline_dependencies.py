# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CMAKE = shutil.which("cmake")
MAKESELF_ARCHIVE = "makeself-release-2.5.0-patch1.tar.gz"


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "source with spaces"
    shutil.copytree(REPO_ROOT / "csrc/cmake/third_party", source / "cmake/third_party")
    if sys.platform == "darwin":
        # GNU patch on the target Linux server strips CRLF in these upstream
        # diffs. BSD patch does not, so supply equivalent LF diffs on macOS.
        for patch in (source / "cmake/third_party/build/modules/patch").glob("*.patch"):
            patch.write_bytes(patch.read_bytes().replace(b"\r\n", b"\n"))
    return source


def configure(source, content):
    if CMAKE is None:
        pytest.skip("CMake is required for real dependency configuration tests")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(offline_dependencies LANGUAGES NONE)\n"
        'set(CANN_3RD_LIB_PATH "${PROJECT_SOURCE_DIR}/cache with spaces")\n'
        'set(CANN_3RD_PKG_PATH "${CANN_3RD_LIB_PATH}/pkg")\n' + content
    )
    env = os.environ.copy()
    # Local archives must work even when any accidental network request fails.
    env.update(HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1")
    return subprocess.run(
        [CMAKE, "-S", str(source), "-B", str(source / "output")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def build(source, target):
    env = os.environ.copy()
    env.update(HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1")
    return subprocess.run(
        [CMAKE, "--build", str(source / "output"), "--target", target],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def include(name):
    return f'include("${{PROJECT_SOURCE_DIR}}/cmake/third_party/{name}")\n'


def test_makeself_restores_to_actual_cache_path_and_is_reusable(source_tree):
    result = configure(source_tree, include("makeself-fetch.cmake"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Using bundled build dependency: {MAKESELF_ARCHIVE}" in result.stdout
    cache = source_tree / "cache with spaces/makeself"
    for name in ("makeself.sh", "makeself-header.sh"):
        assert (cache / name).stat().st_mode & stat.S_IXUSR
    version = subprocess.run(["bash", str(cache / "makeself.sh"), "--version"], capture_output=True, text=True)
    assert version.returncode == 0
    assert "2.5.0" in version.stdout

    (source_tree / "cmake/third_party/archives" / MAKESELF_ARCHIVE).unlink()
    repeat = configure(source_tree, include("makeself-fetch.cmake"))
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    assert "Preparing makeself" not in repeat.stdout


def test_makeself_repairs_scripts_left_nonexecutable_by_failed_configure(source_tree):
    cache = source_tree / "cache with spaces/makeself"
    cache.mkdir(parents=True)
    for name in ("makeself.sh", "makeself-header.sh"):
        script = cache / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o600)
    result = configure(source_tree, include("makeself-fetch.cmake"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preparing makeself" not in result.stdout
    assert os.access(cache / "makeself.sh", os.X_OK)
    assert os.access(cache / "makeself-header.sh", os.X_OK)


@pytest.mark.parametrize("failure", ["corrupt", "missing_manifest", "missing_checksum", "duplicate_checksum"])
def test_invalid_bundles_fail_during_configure(source_tree, failure):
    archives = source_tree / "cmake/third_party/archives"
    manifest = archives / "SHA256SUMS"
    if failure == "corrupt":
        with (archives / MAKESELF_ARCHIVE).open("ab") as stream:
            stream.write(b"transfer corruption")
        expected = "SHA256 mismatch"
    elif failure == "missing_manifest":
        manifest.unlink()
        expected = "checksum manifest"
    elif failure == "missing_checksum":
        manifest.write_text(
            "".join(line for line in manifest.read_text().splitlines(True) if MAKESELF_ARCHIVE not in line)
        )
        expected = "Missing or invalid build dependency SHA256"
    else:
        line = next(line for line in manifest.read_text().splitlines(True) if MAKESELF_ARCHIVE in line)
        with manifest.open("a") as stream:
            stream.write(line)
        expected = "Duplicate build dependency checksum"
    result = configure(source_tree, include("makeself-fetch.cmake"))
    assert result.returncode != 0
    assert expected in result.stdout + result.stderr


def test_archive_absence_keeps_pinned_download_fallback(source_tree):
    (source_tree / "cmake/third_party/archives" / MAKESELF_ARCHIVE).unlink()
    content = include("offline_archive.cmake") + (
        f'vllm_ascend_dependency_url(URL "{MAKESELF_ARCHIVE}" "https://example.invalid/archive.tar.gz")\n'
        'file(WRITE "${PROJECT_BINARY_DIR}/selected.txt" "${URL}\\n${URL_SHA256}")\n'
    )
    result = configure(source_tree, content)
    assert result.returncode == 0, result.stdout + result.stderr
    url, digest = (source_tree / "output/selected.txt").read_text().splitlines()
    assert url == "https://example.invalid/archive.tar.gz"
    assert digest == "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a"


def test_json_installs_headers_without_network(source_tree):
    result = configure(source_tree, include("json.cmake"))
    assert result.returncode == 0, result.stdout + result.stderr
    result = build(source_tree, "third_party_json")
    assert result.returncode == 0, result.stdout + result.stderr
    header = source_tree / "cache with spaces/json/include/nlohmann/json.hpp"
    assert header.is_file()
    macros = header.parent / "detail/abi_macros.hpp"
    assert "NLOHMANN_JSON_VERSION_PATCH 3" in macros.read_text()


def test_json_reuses_preinstalled_headers(source_tree):
    header = source_tree / "cache with spaces/json/include/nlohmann/json.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("// preinstalled header\n")
    (source_tree / "cmake/third_party/archives/include.zip").unlink()
    result = configure(source_tree, include("json.cmake"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not force rebuild cann third_party" in result.stdout + result.stderr
    assert header.read_text() == "// preinstalled header\n"


def test_protobuf_and_abseil_download_and_apply_existing_patches_offline(source_tree):
    content = (
        include("abseil-cpp.cmake")
        + include("ascend_protobuf.cmake")
        + (
            "ExternalProject_Add_StepTargets(abseil_build_transformer patch)\n"
            "ExternalProject_Add_StepTargets(ascend_protobuf_build_transformer patch)\n"
        )
    )
    result = configure(source_tree, content)
    assert result.returncode == 0, result.stdout + result.stderr
    for target in ("abseil_build_transformer-patch", "ascend_protobuf_build_transformer-patch"):
        result = build(source_tree, target)
        assert result.returncode == 0, result.stdout + result.stderr
    abseil = source_tree / "third_party/abseil-cpp/absl/base/options.h"
    assert "lts_ascend_private" in abseil.read_text()
    protobuf = source_tree / "third_party/ascend_protobuf/CMakeLists.txt"
    assert 'set(protobuf_VERSION_STRING "3.13.0")' in protobuf.read_text()
    assert "ArenaStringPtr::Set(EmptyDefault" in (protobuf.parent / "src/google/protobuf/arenastring.cc").read_text()


@pytest.mark.parametrize("directory_exists", [True, False])
def test_prepare_does_not_run_make_after_failed_configuration(tmp_path, directory_exists):
    # The real prepare script has no `set -e`; test its function in that context.
    source = (REPO_ROOT / "csrc/cmake/scripts/prepare.sh").read_text()
    function = re.search(r"(?ms)^function build\(\) \{.*?^\}", source)
    assert function is not None
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls"
    for name, status in (("cmake", 42), ("make", 0)):
        script = tools / name
        script.write_text(f"#!/bin/sh\nprintf '%s\\n' {name} >> {shlex.quote(str(calls))}\nexit {status}\n")
        script.chmod(0o755)
    output = tmp_path / "output with spaces"
    if directory_exists:
        output.mkdir()
    harness = tmp_path / "prepare.sh"
    harness.write_text('PATH_TO_BUILD="$1"\nPATH_TO_SOURCE="$2"\nJOB_NUM=-j1\n' + function.group() + "\nbuild\n")
    env = os.environ.copy()
    env["PATH"] = str(tools) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(harness), str(output), str(tmp_path)], env=env, capture_output=True)
    assert result.returncode == (42 if directory_exists else 1)
    assert (calls.read_text().splitlines() if calls.exists() else []) == (["cmake"] if directory_exists else [])
