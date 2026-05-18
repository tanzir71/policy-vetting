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

## Benchmark Results

Two owner-run notebook benchmarks were used to judge the adapter.

The first run showed that the model had learned some Bangladesh-specific
vocabulary and JSON shapes, but it was not reliable enough for a demo:

- valid JSON was only `3 / 8`
- disclaimer/refusal behavior appeared in only `2 / 8`
- several generations were truncated or malformed
- company setup, expansion, and commercial-contract answers sometimes grounded
  themselves in unrelated excerpts
- one commercial/vendor answer cited the Negotiable Instruments Act for a
  broader contract-vetting task, which is not useful for end users

After the notebook repair pass, the second benchmark produced:

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

The repaired benchmark covered these probe categories:

| Probe | Observed behavior after repair |
|---|---|
| Unsupported authority prediction | Refused to predict a court/government outcome and offered a safer source-summary/checklist path. |
| Source/task mismatch | Correctly refused to use a Negotiable Instruments excerpt as the basis for a company setup pathway. |
| Company setup | Returned source-supported setup points, broader checks requiring other sources, missing facts, citations, and disclaimer. |
| Partnership/JV | Returned agreement drafting points, missing facts, Partnership Act citation, and expert-review framing. |
| Expansion pathway | Flagged Bangladesh Bank/foreign-exchange relevance when the supplied source supported it and separated tax, licensing, BIDA/BEPZA/BEZA, employee-transfer, and stamp-duty checks as requiring other sources. |
| Commercial contract vetting | Produced a vendor/supply/service-contract checklist around scope, acceptance, payment, warranty/support, liability, termination, records, dispute terms, and citations. |
| Company policy vetting | Flagged blanket "all sales final/no refund/no warranty/no support/no escalation" language as review-required and suggested a clearer refund/return/warranty/service/complaint policy structure. |
| Disciplinary timeline | Returned JSON and warned not to proceed without expert review, but the selected benchmark source was still legally adjacent rather than ideal labor-law grounding. |
| Benchmark alignment | Returned the intended assistant rules: source-grounded, cautious, checklist/redline oriented, no invented approvals or predictions. |

The important improvement is not just formatting. The second benchmark shows
the adapter can now produce structured, source-aware exploration outputs across
company setup, partnership/JV, expansion, commercial contract, refund/warranty/
service policy, and labor-process probes without falling out of JSON.

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
or substitute for a qualified Bangladeshi professional.

The main remaining weakness is source selection, not JSON compliance. In the
latest benchmark, the disciplinary-timeline probe produced a cautious answer but
was grounded in a Contract Act excerpt instead of an ideal Labour Act/Rules
excerpt. Production systems should therefore use retrieval, task/source filters,
JSON validation, and human review before presenting outputs to end users.

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
