"""
push_to_hub.py

Upload train/validation splits for the Bangladesh contract, labor, and policy
vetting dataset to a uniquely named Hugging Face dataset repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

DEFAULT_REPO = "YOUR_HF_USER/bd-contract-labor-policy-vetting-live-sft"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s push_to_hub %(message)s",
)
log = logging.getLogger("push_to_hub")


def count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def task_counts(path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task_type", "unknown")
            counts[task] = counts.get(task, 0) + 1
    return counts


def dataset_card(train: str, val: str) -> str:
    train_n = count_jsonl(train)
    val_n = count_jsonl(val)
    counts = task_counts(train)
    for task, n in task_counts(val).items():
        counts[task] = counts.get(task, 0) + n
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""---
license: other
language:
- en
- bn
tags:
- legal
- bangladesh
- labor-law
- contract-vetting
- consumer-policy
- warranty
- refund-policy
- service-policy
- company-policy
- foreign-investment
- epz
- qlora
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# Bangladesh Contract, Labor, and Policy Vetting Live SFT

Generated on {generated}.

Rows:
- train: {train_n}
- validation: {val_n}

Task counts:

```json
{json.dumps(counts, indent=2, ensure_ascii=False)}
```

This full-size snapshot is built from bounded live-source harvesting of
Bangladesh legal sources, BEPZA/EPZ policy sources, and BPPA/CPTU procurement
sources where reachable. It includes source-grounded clause vetting, redlines,
fact intake, compliance checklists, expert handoff packets, benchmark-alignment
examples, company setup/JV/expansion rows, labor/HR rows, and refund/warranty/
service/company-policy rows. The builder applies source/task relevance guards
so, for example, company setup and customer-policy examples are not paired with
unrelated negotiable-instrument excerpts.

It is intended for model training in exploration and drafting support workflows
only. It is not legal advice and should not be used without source verification
and qualified expert review.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--source-card", default="data/source_manifest_card.md")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.repo_id.startswith("YOUR_HF_USER/"):
        log.error("replace --repo-id with your Hugging Face namespace")
        return 2
    train_n = count_jsonl(args.train)
    val_n = count_jsonl(args.val)
    if train_n == 0 or val_n == 0:
        log.error("empty train/val splits (train=%d val=%d)", train_n, val_n)
        return 1

    try:
        from huggingface_hub import create_repo, upload_folder
    except ImportError as exc:
        log.error("huggingface_hub is required for upload. Install requirements.txt first.")
        return 3

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(args.train, os.path.join(tmp, "train.jsonl"))
        shutil.copy2(args.val, os.path.join(tmp, "val.jsonl"))
        if os.path.exists(args.source_card):
            shutil.copy2(args.source_card, os.path.join(tmp, "SOURCE_MANIFEST.md"))
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write(dataset_card(args.train, args.val))
        upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=tmp,
            token=args.token,
            commit_message=f"Upload live-source SFT splits train={train_n} val={val_n}",
        )
    print(f"OK https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
