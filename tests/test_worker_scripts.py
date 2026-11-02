"""Shell-level tests for scripts/run-worker.sh and scripts/pre-push-verify.sh (#19).

Both scripts are exercised for real in subprocesses, with fakes standing in
for the venv interpreter, the worker binary, ``uv`` and ``make`` — nothing
here touches the live venv, the launchd worker, or the library DB.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_WORKER = REPO / "scripts" / "run-worker.sh"
PRE_PUSH = REPO / "scripts" / "pre-push-verify.sh"

# Scrub git hook env so scratch repos aren't redirected into the real one.
_GIT_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _clean_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_VARS}
    env.update(extra)
    return env


# ─── run-worker.sh ───────────────────────────────────────────────────────────


@pytest.fixture
def worker_env(tmp_path):
    marker = tmp_path / "worker-ran"
    home = tmp_path / "home"
    home.mkdir()
    worker = _executable(tmp_path / "fake-worker", f'echo "$@" > "{marker}"\n')

    def make(import_ok: bool):
        py = _executable(tmp_path / "fake-python", "exit 0\n" if import_ok else "exit 1\n")
        return _clean_env(
            HOME=str(home),
            WORKER_PYTHON=str(py),
            WORKER_BIN=str(worker),
            WORKER_GUARD_SLEEP="0",
        )

    return make, marker


def _run_worker(env):
    return subprocess.run(["bash", str(RUN_WORKER)], env=env, capture_output=True, text=True, timeout=30)


def test_run_worker_missing_package_fails_loudly_and_does_not_exec(worker_env):
    make, marker = worker_env
    result = _run_worker(make(import_ok=False))

    assert result.returncode != 0
    assert not marker.exists(), "worker must not start without book_ingestion"
    assert "book_ingestion" in result.stderr
    assert "uv pip install -e" in result.stderr
    assert "--no-deps" in result.stderr
    assert "launchctl kickstart" in result.stderr


def test_run_worker_missing_package_message_is_one_line(worker_env):
    make, _ = worker_env
    result = _run_worker(make(import_ok=False))
    assert len(result.stderr.strip().splitlines()) == 1


def test_run_worker_package_present_execs_worker(worker_env):
    make, marker = worker_env
    result = _run_worker(make(import_ok=True))

    assert result.returncode == 0, result.stderr
    assert marker.read_text().strip() == "worker"


# ─── pre-push-verify.sh ──────────────────────────────────────────────────────


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def hook_repo(tmp_path):
    """A scratch main checkout + a linked worktree, with stub uv/make on PATH."""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _git(main, "config", "user.email", "t@example.com")
    _git(main, "config", "user.name", "t")
    (main / "scripts").mkdir()
    shutil.copy(PRE_PUSH, main / "scripts" / "pre-push-verify.sh")
    _git(main, "add", "scripts")
    _git(main, "commit", "-q", "-m", "init")
    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(linked))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _executable(bin_dir / "uv", f'echo "uv $@" >> "{log}"\n')
    _executable(bin_dir / "make", f'echo "make $@" >> "{log}"\n')
    return main, linked, bin_dir, log


def _run_hook(checkout, bin_dir):
    env = _clean_env(PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(checkout / "scripts" / "pre-push-verify.sh")],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_hook_main_checkout_does_not_sync_venv(hook_repo):
    main, _, bin_dir, log = hook_repo
    result = _run_hook(main, bin_dir)

    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "uv sync" not in calls, "main checkout's venv must not be mutated"
    assert "make lint" in calls and "make test" in calls
    assert "skipping uv sync" in result.stdout


def test_hook_linked_worktree_still_syncs_venv(hook_repo):
    _, linked, bin_dir, log = hook_repo
    # The worktree carries its own copy of the script (same tracked file).
    result = _run_hook(linked, bin_dir)

    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "uv sync --locked --python 3.12 --extra dev" in calls
    assert "make lint" in calls and "make test" in calls
