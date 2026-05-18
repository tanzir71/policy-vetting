# Bangladesh Contract, Labor, and Policy Vetting Dataset

This folder contains a live-source-backed SFT pipeline for training a
Qwen-style LLM that helps **any business operating or planning to
operate in Bangladesh** explore contract, labor, and policy compliance
issues before engaging a qualified expert.

The assistant is built to be general purpose. It covers:

- new company setup (incorporation with the Registrar of Joint Stock
  Companies and Firms, BIDA registration, trade licence, TIN, BIN/VAT)
- partnerships, joint ventures, and shareholders' agreements (local-local,
  local-foreign, and family-business arrangements)
- expansion for existing businesses (branch and subsidiary opening, M&A,
  restructuring, share allotment, capital raises)
- routine commercial contracts (vendor and supply, services, distribution,
  NDA, IP assignment and licensing, inter-company agreements)
- customer-facing company policies (refund/return, service policy,
  warranty/guarantee, complaint handling, privacy/data terms)
- general labor and HR policy under the Bangladesh Labour Act, 2006
- specialised regimes: EPZ/BEPZA operations, government procurement
  (BPPA/CPTU), and foreign investment

Foreigners and expat-led ventures are covered as one important audience -
but the dataset is no longer tilted toward them. The same model is
expected to be useful to a local family business formalising its books,
a Dhaka-based SME signing a vendor contract, two existing companies
forming a joint venture, or a foreign investor exploring entry options.

The system is for exploration and drafting support only. It is not a
lawyer, immigration adviser, tax adviser, or substitute for a Bangladeshi
advocate. Training rows push the model to cite sources, flag uncertainty,
and ask for missing facts.

## What This Builds

- `data/manifest.json` - live-source provenance with retrieval timestamps.
- `data/source_chunks.jsonl` - extracted source chunks from official sources.
- `data/dataset.jsonl` - full source-grounded SFT rows across the task types below.
- `data/train.jsonl` and `data/val.jsonl` - stratified training splits.
- `colab_train_bd_contract_labor_policy_qwen_live_sources.ipynb` - the
  Colab QLoRA notebook.

## Live Sources

The collector is deliberately bounded but real:

- **Laws of Bangladesh portal** - core statutes: Contract Act 1872,
  Sale of Goods Act 1930, Companies Act 1994, Bangladesh Labour Act
  2006, Foreign Private Investment (Promotion and Protection) Act 1980,
  Consumer Rights Protection Act 2009, Partnership Act 1932,
  Negotiable Instruments Act 1881, Arbitration Act 2001, Specific
  Relief Act 1877, and Foreign Exchange Regulation Act 1947. These
  support contract, warranty, refund/return, customer complaint, service
  policy, company, labor, foreign-investment, partnership, payment,
  arbitration, remedy, and cross-border vetting.
- **BPPA/CPTU standard tender documents** - a small selection of
  procurement and service contracts. Used as a specialisation.
- **BEPZA Acts, Policies, SROs** - EPZ Labour Act/Rules, work permit
  policy, EPZ wage/inspection materials, and foreign investment/OSS
  links when discoverable. Used as a specialisation.

Run the collector again whenever you want fresher retrieval timestamps
and newly published linked documents.

The current full snapshot was refreshed on 2026-05-17 from 20 live
records and 1,699 source chunks: 11 Laws of Bangladesh act-print pages
(Contract Act, Sale of Goods Act, Companies Act, Labour Act, Foreign
Private Investment Act, Consumer Rights Protection Act, Partnership Act,
Negotiable Instruments Act, Arbitration Act, Specific Relief Act, and
Foreign Exchange Regulation Act), plus the BEPZA Acts and Policies index
and 8 linked BEPZA PDFs. CPTU remains enabled in the collector but was
DNS-unreachable during this run; the failure is logged in
`data/failed_sources.log`.

Some Bangla PDFs use CID-encoded fonts that extract as `(cid:...)`
artifacts. The collector filters those noisy chunks and keeps usable
extracted text, such as the English EPZ Labour Act, while still
retaining PDF provenance in `data/manifest.json`.

The current full build is quality-filtered rather than raw-expanded:
44,055 rows from 1,699 live-source chunks. The builder now rejects major
source/task mismatches, such as using a Negotiable Instruments excerpt to
teach company setup, commercial contract, expansion, partnership/JV, or
refund/warranty policy behavior.

## Quick Start

```powershell
cd C:\Users\tanzir\Desktop\contract-labor-policy-vetting\bd-contract-labor-policy-vetting
python -m pip install -r requirements.txt
python collect_live_sources.py --max-bepza-links 6 --max-cptu-docs 6 --max-bdlaws-acts 9 --include-candidate-acts --delay 1
python build_vetting_dataset.py --val-frac 0.15 --profile full --expansion-depth 3
python validate_artifacts.py --min-dataset-rows 40000 --min-task-count company_policy_vetting=2000 --min-task-count commercial_contract_vetting=1500 --min-task-count company_setup_pathway=150
```

For a smaller smoke test:

```powershell
python collect_live_sources.py --max-bepza-links 4 --max-cptu-docs 2 --max-bdlaws-acts 4 --max-pdf-pages 10 --delay 0.5
python build_vetting_dataset.py --max-rows 240
python validate_artifacts.py
```

## Hugging Face Upload

The default dataset repo name is intentionally unique so it does not
overwrite your existing legal/Qwen assets:

```powershell
$env:HF_TOKEN="<paste-token-here>"
python push_to_hub.py --repo-id <your-user>/bd-contract-labor-policy-vetting-live-sft --private
```

To publish or refresh the Hugging Face model card for the trained adapter:

```powershell
$env:HF_TOKEN="<paste-token-here>"
python push_model_card.py --repo-id tanziro/bd-contract-labor-policy-vetting-qwen25-3b-lora-json-grounded-repair --private
```

The Colab notebook defaults to:

- dataset: `<hf-user>/bd-contract-labor-policy-vetting-live-sft`
- repair source adapter: `<hf-user>/bd-contract-labor-policy-vetting-qwen25-3b-lora`
- repaired adapter output: `<hf-user>/bd-contract-labor-policy-vetting-qwen25-3b-lora-json-grounded-repair`

Change `HF_USER` in the notebook if you want an org namespace.

The notebook now defaults to `SKIP_TRAINING = True` so a fresh Colab
upload followed by **Run all** loads the repaired adapter and runs the
sanity probes without starting paid training. Set `SKIP_TRAINING = False`
only when you intentionally want to train or continue the repair pass.

When training is enabled, `QUALITY_REPAIR_MODE = True` starts from the
already trained adapter, writes to a new Hub repo, filters out source/task
mismatches, injects repeated benchmark anchors, and trains roughly 9k
high-signal rows for one epoch. That should be far cheaper than another
full run while targeting the observed failures: invalid/truncated JSON and
generic checklists grounded in unrelated source excerpts. Set
`QUALITY_REPAIR_MODE = False` only when you want a fresh half-day run from
the base Qwen model.

The notebook also has a paid-run persistence preflight. It mounts Google
Drive, writes and reads a Drive probe, creates the Hugging Face model
repo, uploads a Hub probe, and refuses to train unless it prints
`PERSISTENCE PREFLIGHT PASSED`. Trainer checkpoints are written under
`DRIVE_BACKUP_DIR/trainer-output/checkpoint-*`, adapter checkpoints are
saved to Google Drive under `DRIVE_BACKUP_DIR/checkpoint-*`, and the same
adapter checkpoints are uploaded to Hugging Face under
`adapter-checkpoints/checkpoint-*`. If Colab disconnects, rerunning the
notebook resumes from the latest Drive Trainer checkpoint when available.

## Dataset Schema

Each JSONL row follows the existing `legal-assistant` style:

```text
instruction, context, reasoning, response, citations,
source_title, source_url, source_type, jurisdiction,
topic, task_type, confidence, refusal_reason
```

Task types are split between general-purpose business vetting and
specialised regimes. General-purpose tasks now carry the majority of the
training weight:

General purpose:

- `clause_vetting`
- `redline_suggestion`
- `source_grounded_summary` (new) - concise source role and limits
- `compliance_checklist` (new) - practical pre-lawyer compliance checklist
- `fact_intake_triage` (new) - missing-fact questions before applying law
- `clause_comparison` (new) - safer clause selection and caveats
- `expert_handoff_packet` (new) - documents/questions for professional review
- `benchmark_alignment` (new) - trains Bangladesh-specific, source-grounded benchmark behavior
- `company_setup_pathway` (new) - incorporation, RJSC, governance, filings
- `partnership_jv_vetting` (new) - partnership deeds, JV and shareholders' agreements
- `expansion_pathway` (new) - branch/subsidiary opening, M&A, restructuring
- `commercial_contract_vetting` (new) - vendor, services, NDA, IP, distribution
- `company_policy_vetting` (new) - refunds, returns, warranty/guarantee, service policy, privacy, complaints
- `general_employment_vetting` (new) - appointment letters, leave, wage, gratuity
- `disciplinary_timeline_check`
- `clarification`
- `bilingual_term_mapping`
- `refusal`

Specialised:

- `epz_applicability`
- `foreign_investor_orientation`
- `procurement_contract_architecture`

Synthetic clauses and scenarios are generated only as instruction
wrappers around retrieved source chunks. The source text remains the
legal ground truth.

## Safety Posture

- Every substantive row includes citations.
- Responses are framed as issue spotting and exploration, not final
  legal advice.
- The model is trained to ask for missing facts before applying law to
  a user situation.
- Unsupported prediction and personalized advice prompts are refusal rows.
- End users should verify against official portals and hire a qualified
  Bangladeshi expert before acting.
