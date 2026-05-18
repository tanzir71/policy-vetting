"""
validate_artifacts.py

Local sanity checks for generated source, dataset, split, and notebook files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


REQUIRED_ROW_FIELDS = {
    "instruction",
    "context",
    "reasoning",
    "response",
    "citations",
    "source_title",
    "source_url",
    "source_type",
    "jurisdiction",
    "topic",
    "task_type",
    "confidence",
    "refusal_reason",
}

NOTEBOOK_NAME = "colab_train_bd_contract_labor_policy_qwen_live_sources.ipynb"
EXPECTED_DATASET_NAME = "bd-contract-labor-policy-vetting-live-sft"
EXPECTED_MODEL_NAME = "bd-contract-labor-policy-vetting-qwen25-3b-lora"


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise AssertionError(f"missing file: {path}")
    if os.path.getsize(path) == 0:
        raise AssertionError(f"empty file: {path}")


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(term.lower() in lowered for term in terms)


def validate_source_alignment(row: dict, name: str, idx: int) -> None:
    task = row.get("task_type")
    title = row.get("source_title", "")
    if "negotiable instruments" in title.lower() and task in {
        "company_setup_pathway",
        "partnership_jv_vetting",
        "expansion_pathway",
        "commercial_contract_vetting",
        "company_policy_vetting",
    }:
        raise AssertionError(f"{name} row {idx} pairs {task} with Negotiable Instruments source")
    if task == "company_setup_pathway" and not has_any(title, ("companies", "কোম্পানী")):
        raise AssertionError(f"{name} row {idx} company setup has non-company source: {title}")
    if task == "commercial_contract_vetting" and not has_any(
        title,
        ("contract act", "sale of goods", "arbitration", "সালিস", "specific relief", "partnership act"),
    ):
        raise AssertionError(f"{name} row {idx} commercial contract has weak source: {title}")
    if task == "company_policy_vetting" and not has_any(
        title,
        ("companies", "কোম্পানী", "বাংলাদেশ শ্রম আইন", "consumer", "ভোক্তা", "sale of goods", "contract act"),
    ):
        raise AssertionError(f"{name} row {idx} company policy has weak source: {title}")


def validate_rows(rows: list[dict], name: str) -> Counter:
    if not rows:
        raise AssertionError(f"{name} has no rows")
    counts: Counter = Counter()
    for idx, row in enumerate(rows):
        missing = REQUIRED_ROW_FIELDS - set(row)
        if missing:
            raise AssertionError(f"{name} row {idx} missing {sorted(missing)}")
        if not row["instruction"].strip():
            raise AssertionError(f"{name} row {idx} has empty instruction")
        if not row["context"].strip():
            raise AssertionError(f"{name} row {idx} has empty context")
        if not row["response"].strip():
            raise AssertionError(f"{name} row {idx} has empty response")
        if row["task_type"] != "refusal" and not row.get("citations"):
            raise AssertionError(f"{name} row {idx} is non-refusal without citations")
        validate_source_alignment(row, name, idx)
        counts[row["task_type"]] += 1
    return counts


def validate_notebook(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    blob = json.dumps(notebook)
    if EXPECTED_DATASET_NAME not in blob:
        raise AssertionError(f"notebook does not mention {EXPECTED_DATASET_NAME}")
    if EXPECTED_MODEL_NAME not in blob:
        raise AssertionError(f"notebook does not mention {EXPECTED_MODEL_NAME}")
    persistence_markers = [
        "PERSISTENCE PREFLIGHT PASSED",
        "DRIVE_BACKUP_DIR",
        "AdapterPersistenceCallback",
        "adapter-checkpoints",
        "resume_from_checkpoint=resume_from",
        "FINAL_HUB_SUBFOLDER",
        "QUALITY_REPAIR_MODE",
        "SKIP_TRAINING",
        "ADAPTER_TO_LOAD",
        "quality_row",
        "json-grounded-repair",
        "source_does_not_support_requested_task",
    ]
    for marker in persistence_markers:
        if marker not in blob:
            raise AssertionError(f"notebook missing persistence marker: {marker}")
    if notebook.get("nbformat") != 4:
        raise AssertionError("notebook nbformat is not 4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--notebook", default=NOTEBOOK_NAME)
    parser.add_argument("--min-dataset-rows", type=int, default=0)
    parser.add_argument(
        "--min-task-count",
        action="append",
        default=[],
        metavar="TASK=N",
        help="Require at least N rows for a task type. Can be passed multiple times.",
    )
    args = parser.parse_args()

    manifest = os.path.join(args.data_dir, "manifest.json")
    chunks = os.path.join(args.data_dir, "source_chunks.jsonl")
    dataset = os.path.join(args.data_dir, "dataset.jsonl")
    train = os.path.join(args.data_dir, "train.jsonl")
    val = os.path.join(args.data_dir, "val.jsonl")
    for path in (manifest, chunks, dataset, train, val, args.notebook):
        require_file(path)

    with open(manifest, "r", encoding="utf-8") as f:
        manifest_obj = json.load(f)
    if not manifest_obj.get("records"):
        raise AssertionError("manifest has no records")
    if not read_jsonl(chunks):
        raise AssertionError("source chunks file has no chunks")

    dataset_rows = read_jsonl(dataset)
    train_rows = read_jsonl(train)
    val_rows = read_jsonl(val)
    dataset_counts = validate_rows(dataset_rows, "dataset")
    validate_rows(train_rows, "train")
    validate_rows(val_rows, "val")
    validate_notebook(args.notebook)
    if args.min_dataset_rows and len(dataset_rows) < args.min_dataset_rows:
        raise AssertionError(f"dataset has {len(dataset_rows)} rows, below required {args.min_dataset_rows}")
    for item in args.min_task_count:
        if "=" not in item:
            raise AssertionError(f"--min-task-count must be TASK=N, got {item!r}")
        task, raw_min = item.split("=", 1)
        try:
            required = int(raw_min)
        except ValueError as exc:
            raise AssertionError(f"invalid minimum for {task!r}: {raw_min!r}") from exc
        actual = dataset_counts.get(task, 0)
        if actual < required:
            raise AssertionError(f"task {task!r} has {actual} rows, below required {required}")

    print(json.dumps({
        "records": len(manifest_obj["records"]),
        "chunks": len(read_jsonl(chunks)),
        "dataset_rows": len(dataset_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "task_counts": dict(dataset_counts),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
