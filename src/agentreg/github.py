"""Post the report as one sticky PR comment that is edited on every push."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from agentreg.report import MARKER

API = "https://api.github.com"
MAX_BODY = 65_000  # GitHub's limit is 65,536 characters


def pr_from_env() -> tuple[str, int]:
    """Read the repository and PR number from the GitHub Actions environment."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not repo or not event_path:
        raise RuntimeError("Not running in GitHub Actions; pass --repo and --pr explicitly.")
    event = json.loads(Path(event_path).read_text())
    number = (event.get("pull_request") or {}).get("number") or event.get("number")
    if not number:
        raise RuntimeError("This workflow event is not a pull request.")
    return repo, int(number)


def upsert_comment(body: str, repo: str, pr: int, token: str, client: httpx.Client | None = None) -> str:
    """Edit our existing comment if there is one, otherwise create it. Returns its URL."""
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + "\n\n… report truncated; see the workflow artifact for the full report."
    if MARKER not in body:
        body = f"{MARKER}\n{body}"

    own_client = client is None
    client = client or httpx.Client(
        base_url=API,
        timeout=30,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        existing = None
        page = 1
        while existing is None:
            resp = client.get(f"/repos/{repo}/issues/{pr}/comments", params={"per_page": 100, "page": page})
            resp.raise_for_status()
            comments = resp.json()
            existing = next((c for c in comments if MARKER in (c.get("body") or "")), None)
            if len(comments) < 100:
                break
            page += 1

        if existing:
            resp = client.patch(f"/repos/{repo}/issues/comments/{existing['id']}", json={"body": body})
        else:
            resp = client.post(f"/repos/{repo}/issues/{pr}/comments", json={"body": body})
        resp.raise_for_status()
        return resp.json()["html_url"]
    finally:
        if own_client:
            client.close()
