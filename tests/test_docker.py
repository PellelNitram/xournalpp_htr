"""Tests for the Hugging Face demo Docker image.

All tests are marked ``slow``. Run them with::

    uv run pytest -m slow tests/test_docker.py -v

Prerequisites: Docker daemon must be running.
"""

import logging
import urllib.request
from pathlib import Path

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient
from testcontainers.core.waiting_utils import wait_for_logs

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_NAME = "xournalpp_htr_test"

# Dummy values that satisfy demo.py's get_env_variable() calls at import
# time without requiring real Supabase credentials.  The Supabase client
# itself is only constructed inside request handlers, so the server starts
# fine with these placeholders.
_DUMMY_ENV = {
    "DEMO": "1",
    "SB_URL": "https://dummy.supabase.co",
    "SB_KEY": "dummy-key",
    "SB_BUCKET_NAME": "dummy-bucket",
    "SB_SCHEMA_NAME": "public",
    "SB_TABLE_NAME": "dummy-table",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def docker_image():
    """Build the Docker image once per test session and return its tag."""
    client = DockerClient()
    _image, logs = client.client.images.build(
        path=str(PROJECT_ROOT), tag=IMAGE_NAME, rm=True
    )
    for event in logs:
        line = event.get("stream", "").rstrip()
        if line:
            log.info(line)
    log.info("Built image: %s", _image)
    return IMAGE_NAME


@pytest.fixture()
def demo_container(docker_image):
    """Start the demo container with dummy env vars; stop it after the test."""
    container = DockerContainer(docker_image)
    for key, value in _DUMMY_ENV.items():
        container = container.with_env(key, value)
    container = container.with_exposed_ports(7860)
    with container:
        wait_for_logs(container, "Running on", timeout=90)
        yield container


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_docker_image_builds(docker_image):
    """The Docker image builds without error."""


@pytest.mark.slow
def test_docker_xournalpp_available(docker_image):
    """The xournalpp binary is installed and on PATH inside the container."""
    client = DockerClient()
    output = client.client.containers.run(
        docker_image,
        command=["which", "xournalpp"],
        remove=True,
    )
    assert b"xournalpp" in output


@pytest.mark.slow
def test_docker_hf_packages_importable(docker_image):
    """All HF-extra Python packages are importable inside the container."""
    client = DockerClient()
    output = client.client.containers.run(
        docker_image,
        command=[
            "uv",
            "run",
            "python",
            "-c",
            "import gradio, pdf2image, supabase, dotenv; print('OK')",
        ],
        remove=True,
    )
    assert b"OK" in output


@pytest.mark.slow
def test_docker_server_responds(demo_container):
    """The Gradio HTTP server inside the container responds with HTTP 200."""
    port = demo_container.get_exposed_port(7860)
    with urllib.request.urlopen(f"http://localhost:{port}/", timeout=10) as response:
        assert response.status == 200
