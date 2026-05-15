"""Demo/event helpers for the WordDetector HuggingFace Space.

Supabase event logging plus small utilities used by the Gradio demo. The
Supabase/Gradio dependencies come from the ``hf`` extra.
"""

import io
import json
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
from git import Repo


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


def get_example_list() -> List[List]:
    """Return only example images that still exist online."""
    links_to_images = [
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/cvl.jpg",
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/random.jpg",
        "https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/bentham.jpg",
    ]
    return [[link, 0, False] for link in links_to_images if url_exists(link)]


def save_event(
    data: dict,
    SB_URL: str,
    SB_KEY: str,
    SB_SCHEMA_NAME: str,
    SB_TABLE_NAME: str,
    SB_BUCKET_NAME: str,
):
    from supabase import Client, create_client

    supabase: Client = create_client(SB_URL, SB_KEY)

    uid = str(data["uuid"])
    contains_image = data.get("image") is not None

    if contains_image and data["donate_data"]:
        arr = data["image"]
        if not isinstance(arr, np.ndarray):
            raise ValueError("Image must be a numpy array!")

        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        filename = f"{uid}.npy"
        supabase.storage.from_(SB_BUCKET_NAME).upload(
            filename,
            buf.getvalue(),
            {"content-type": "application/octet-stream"},
        )

    row = {
        "timestamp": data["timestamp"].isoformat(),
        "demo": data["demo"],
        "uuid": uid,
        "donate_data": data["donate_data"],
        "contains_image": contains_image,
    }

    supabase.schema(SB_SCHEMA_NAME).table(SB_TABLE_NAME).insert(row).execute()
