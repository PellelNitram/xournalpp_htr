"""Small helpers for WordDetector training and the local demo.

Git-hash capture, a JSON encoder for ``Path``, and the bundled example-image
list. There is no telemetry/Supabase here: per ADR 007 the model ships a
local-only demo with no HuggingFace Space.
"""

import json
import urllib.request
from pathlib import Path
from typing import List

import torch
from git import Repo


def get_device(preference: str = "auto") -> str:
    """Resolve a device string, with auto-detection for ``"auto"``."""
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


class CustomEncoder(json.JSONEncoder):
    """Make non-standard items serialisable for ``json.dump(s)``."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def get_git_commit_hash(repo_path: Path = Path("."), short: bool = False) -> str:
    """Return the current Git commit hash, or ``"-1"`` if unavailable."""
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        commit_hash: str = repo.head.commit.hexsha
        return commit_hash[:7] if short else commit_hash
    except Exception as e:
        print(f"Error while getting git commit hash from {repo_path}: {e}")
        return "-1"


def url_exists(url: str) -> bool:
    """Check if a URL exists using a HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception as e:
        print(e)
        return False


def get_example_list() -> List[str]:
    """Return the bundled example-image URLs that are still reachable."""
    links_to_images = [
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/cvl.jpg",
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/random.jpg",
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/bentham.jpg",
    ]
    return [link for link in links_to_images if url_exists(link)]
