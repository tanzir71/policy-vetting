---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
license: other
language:
- en
- bn
tags:
- qwen
- qwen2.5
- peft
- lora
- qlora
- legal
- bangladesh
- contract-vetting
- company-setup
- labor-law
- company-policy
- refund-policy
- warranty
- service-policy
- foreign-investment
- epz
datasets:
- tanziro/bd-contract-labor-policy-vetting-live-sft
---

# Bangladesh Contract, Labor, and Policy Vetting - Qwen2.5 3B LoRA

This is the **company setup citation repair** adapter for
`Qwen/Qwen2.5-3B-Instruct`. It is a PEFT LoRA trained for source-grounded
exploration of Bangladesh-facing business legal and compliance questions.

The adapter is intended to reduce friction before a business hires an expert.
It helps with early issue spotting, missing-fact intake, checklist generation,
and conservative redline direction for:

- company setup, incorporation, RJSC-oriented checkpoints, and post-incorporation hygiene
- partnerships, joint ventures, and shareholder-agreement review
- expansion paths for existing businesses, including branches, subsidiaries, restructuring, and foreign-exchange touchpoints
- commercial contracts such as vendor, supply, service, NDA, IP, distribution, and support terms
- customer-facing company policies such as refund, return, warranty, service, complaint, privacy, and support policies
- Bangladesh labor and HR policy issue spotting
- EPZ/BEPZA and foreign-investor orientation when the supplied source context supports it

This model is for exploration and preparation, not final legal advice.

## Training Data

The adapter was trained on:

`tanziro/bd-contract-labor-policy-vetting-live-sft`

The dataset is built from live-source-backed Bangladesh legal and business
compliance material, including Laws of Bangladesh act-print pages, BEPZA/EPZ
policy documents, and bounded procurement-related sources when reachable.
Instruction rows are built around source excerpts, citations, source limits,
missing facts, refusal behavior, checklist outputs, and expert handoff language.

## Repair Lineage

This adapter is the latest 3B repair in the sequence:

1. Base Qwen2.5 3B instruct model.
2. Initial Bangladesh contract/labor/policy LoRA.
3. JSON/source-grounding repair.
4. Source-selection repair.
5. Company-setup repair.
6. **Company-setup citation repair**: this repo.

The final repair starts from the company-setup repair adapter and specifically
targets the last observed benchmark issue: a strong company setup answer that
used the right Companies Act incorporation excerpt but sometimes left
`citations` empty.

## Latest Benchmark

The latest owner-run smoke benchmark passed all audited gates without fallback:

```json
{
  "valid_json_raw": 9,
  "valid_json_after_fallback": 9,
  "fallback_used": 0,
  "no_training_tags": 9,
  "has_disclaimer_or_refusal": 9,
  "has_citations_or_refusal": 9,
  "bad_source_task_matches": 0,
  "weak_company_setup_sources": 0,
  "missing_company_setup_citations": 0,
  "total": 9
}
```

Smoke-test result:

- JSON/fallback audit: passed
- source/task matching audit: passed
- company setup source-strength audit: passed
- company setup citation audit: passed
- fallback usage: none

## Probe Coverage

| Probe | Latest observed behavior |
|---|---|
| Unsupported authority prediction | Refused to predict a future court, RJSC, BIDA, BEPZA, NBR, or authority decision and offered safer source-summary/checklist help. |
| Source/task mismatch | Refused to use a Negotiable Instruments Act excerpt as the basis for company setup guidance. |
| Company setup | Used a strong Companies Act section 6 incorporation/memorandum excerpt, separated broader checks, asked missing facts, and included a citation object. |
| Partnership/JV | Returned agreement drafting points, missing facts, Partnership Act citation, and expert-review framing. |
| Expansion pathway | Flagged foreign-exchange/source-supported points while separating tax, licensing, BIDA/BEPZA/BEZA, employee-transfer, and stamp-duty checks as requiring additional sources. |
| Commercial contract vetting | Produced a source-grounded checklist around scope, acceptance, payment, warranty/support, liability, termination, records, dispute terms, and citations. |
| Company policy vetting | Flagged blanket "all sales final/no refund/no warranty/no support/no escalation" language as review-required and suggested a structured refund/return/warranty/service/complaint policy path. |
| Disciplinary timeline | Returned cautious JSON, asked for worker/process facts, separated EPZ/non-EPZ concerns, and warned not to proceed without expert review. |
| Benchmark alignment | Returned the intended assistant rules: Bangladesh-specific, source-grounded, cautious, checklist/redline oriented, no invented approvals or predictions. |

Note: one company-setup probe displayed the Bangla source title as question
marks due to a notebook/display encoding artifact. The citation still retained
the correct source URL, source type, authority, section ID, and chunk ID.

## Intended Use

Suitable prototype outputs include:

- issue-spotting summaries
- source-grounded clause comments
- compliance checklists
- redline direction
- missing-fact questionnaires
- expert handoff packets
- internal demo flows for businesses exploring Bangladesh setup, contracts, labor, or customer-policy compliance

## Usage Example

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_id = "tanziro/bd-contract-labor-policy-vetting-qwen25-3b-lora-company-setup-citation-repair"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_id)
```

Recommended prompt shape:

```text
Vet this Bangladesh-facing company setup, contract, labor, or customer-policy
scenario using only the supplied source excerpt. Return one JSON object with
risk_level, source_supported_points, broader_checks_requiring_additional_sources,
missing_facts_to_confirm, source_grounding, citations, and disclaimer.
```

## Limitations

This adapter does not replace a Bangladeshi advocate, company secretary, tax
professional, immigration adviser, accountant, or regulator-facing expert.

Production use should still add:

- retrieval with task/source filters
- JSON schema validation
- current-law verification against official sources
- user-facing disclaimers
- human review before business, employment, tax, investment, regulatory, or court decisions

Do not use this model to make final legal, employment, investment, tax,
immigration, regulatory, or court strategy decisions.

## Disclaimer

This is automated legal and business-compliance exploration support, not legal
advice. Verify the cited source and consult a qualified Bangladeshi advocate or
relevant professional before acting.
