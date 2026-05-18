"""
Upload the local Hugging Face model card to the trained adapter repository.

The token is read from HF_TOKEN by default. Do not put tokens in this file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL_REPO = "tanziro/bd-contract-labor-policy-vetting-qwen25-3b-lora-json-grounded-repair"


def upload_with_commit_api(repo_id: str, card: str, token: str) -> str:
    """Upload README.md using the Hub commit API without extra dependencies."""
    with open(card, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    payload = "\n".join(
        [
            json.dumps(
                {
                    "key": "header",
                    "value": {
                        "summary": "Update model card",
                        "description": (
                            "Refresh README with intended use, training data, "
                            "limitations, and smoke-test summary."
                        ),
                    },
                }
            ),
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": "README.md",
                        "content": encoded,
                        "encoding": "base64",
                    },
                }
            ),
        ]
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}/commit/main",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-ndjson",
        },
    )
    # Some local Windows Git/Python installs have stale CA config. This helper
    # is only for a small authenticated README upload, not model/data download.
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=context, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hub commit API failed with HTTP {exc.code}: {body}") from exc
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--card", default="model_card.md")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        print("HF_TOKEN is missing. Set it in the environment or pass --token.", file=sys.stderr)
        return 2
    if not os.path.exists(args.card):
        print(f"model card not found: {args.card}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError:
        try:
            body = upload_with_commit_api(args.repo_id, args.card, args.token)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 3
        print(f"OK https://huggingface.co/{args.repo_id}")
        if body:
            print(body[:1000])
        return 0

    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    upload_file(
        path_or_fileobj=args.card,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
        commit_message="Update model card",
    )
    print(f"OK https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
