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

# Bangladesh Contract, Labor, and Policy Vetting Qwen LoRA

This is a PEFT LoRA adapter for `Qwen/Qwen2.5-3B-Instruct`, trained for
source-grounded exploration of Bangladesh-facing business legal and compliance
questions.

The model is intended to reduce friction before a business hires an expert. It
is designed for early issue spotting, missing-fact intake, checklist generation,
and conservative redline direction across:

- company setup, RJSC-oriented incorporation checkpoints, and post-incorporation hygiene
- partnership, JV, and shareholder-agreement review
- expansion paths for existing businesses, including branch, subsidiary, restructuring, and foreign-exchange touchpoints
- commercial contracts such as vendor, supply, service, NDA, IP, distribution, and support terms
- customer-facing company policies such as refund, return, warranty, service, complaint, privacy, and support policies
- Bangladesh labor and HR policy issue spotting
- EPZ/BEPZA and foreign-investor orientation where the source context supports it

## Training Data

The adapter was trained on the live-source SFT dataset:

`tanziro/bd-contract-labor-policy-vetting-live-sft`

The dataset uses bounded live-source harvesting from sources such as Laws of
Bangladesh, BEPZA/EPZ policy sources, and procurement-related sources where
reachable. Rows are instruction-tuning examples built around source excerpts,
citations, source limits, missing facts, and expert handoff language.

## Behavior

The target behavior is:

- return JSON-style, source-grounded answers where requested
- cite the supplied source rather than inventing statutory details
- separate what the excerpt supports from broader checks requiring another source
- ask for missing facts before applying law to a real business decision
- refuse or narrow requests that ask for predictions about courts, regulators, approvals, or outcomes
- avoid treating EPZ and non-EPZ labor rules as interchangeable
- avoid treating refund, warranty, service, privacy, or complaint exclusions as enforceable merely because a company writes them

Recent smoke-test behavior from the notebook:

```json
{
  "valid_json_raw": 9,
  "valid_json_after_fallback": 9,
  "fallback_used": 0,
  "no_training_tags": 9,
  "has_disclaimer_or_refusal": 9,
  "has_citations_or_refusal": 9,
  "total": 9
}
```

## Intended Use

Use this adapter for prototype assistants that help businesses prepare for
professional review. Suitable outputs include:

- issue-spotting summaries
- compliance checklists
- source-grounded clause comments
- redline direction
- missing-fact questionnaires
- expert handoff packets

## Limitations

This model is not a lawyer, tax adviser, immigration adviser, company secretary,
or substitute for a qualified Bangladeshi professional. It may still select an
overbroad or adjacent source when retrieval is weak. Production systems should
use retrieval, source filtering, JSON validation, and human review.

Do not use the model to make final legal, employment, investment, tax,
immigration, regulatory, or court strategy decisions.

## Example Prompt Shape

```text
Vet this Bangladesh-facing refund/warranty policy using only the supplied source.
Return JSON with risk_level, source_supported_checks, missing_facts_to_confirm,
suggested_redline_direction, citations, and disclaimer.
```

## Disclaimer

This is automated legal and business-compliance exploration support, not legal
advice. Verify the cited source and consult a qualified Bangladeshi advocate or
relevant professional before acting.
