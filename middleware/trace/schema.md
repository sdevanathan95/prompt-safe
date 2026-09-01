# Shared trace schema

This is the one contract both tracks must agree on before splitting work.
Every step of the agent's execution produces one of these objects.

```json
{
  "step": 0,
  "source_provenance": "trusted | untrusted",
  "context_label": {
    "integrity": "trusted | untrusted",
    "confidentiality": "public | private"
  },
  "policy_label": {
    "integrity": "trusted | untrusted",
    "confidentiality": "public | private"
  },
  "screened_regions": {
    "relevant": ["region_ids flagged load-bearing by the LM-Judge"],
    "masked": ["region_ids redacted from the agent's context this step"],
    "labels": {
      "region_id": {
        "integrity": "trusted | untrusted",
        "confidentiality": "public | private"
      }
    }
  },
  "policy_verdict": "safe | block | escalate",
  "melon_check": {
    "ran": false,
    "placeholder_task": "the neutral task used in the masked run, if ran",
    "original_calls": [],
    "masked_calls": [],
    "distance": null,
    "verdict": "safe | block | null"
  },
  "final_action": "execute | block | ask_user",
  "explanation": "one or two sentences a human can read to understand why"
}
```

Notes:
- `context_label` is the dependency label: the join of every region the
  screener marked relevant, starting from the most permissive label. It is
  the left-hand side of the policy comparison.
- `policy_label` is `P(tool_call)` — the most restrictive context the policy
  permits this call to be made from. The step's verdict is exactly the
  comparison `context_label ⊑ policy_label`, so both sides are recorded:
  a trace that stores only the outcome can assert a block but cannot explain
  one.
- Labels are a **pair**, not a single axis. The two axes move in opposite
  directions under a join — most restrictive wins for confidentiality, least
  restrictive wins for integrity — so collapsing them loses the
  confidentiality half of the policy entirely.
- `source_provenance` is the integrity axis of `context_label`, duplicated
  for the consumers written against the original single-axis shape. Derived,
  never authoritative; read `context_label` in new code.
- `screened_regions.masked` is not the complement of `relevant`. A region is
  redacted when its own label does not flow to `context_label` — irrelevant
  regions disappear as a consequence of that comparison, not by being
  removed directly. `labels` records each region's own label so the
  redaction decision can be re-derived from the trace.
- `melon_check.ran` is `false` for any step that resolved at the policy-check
  stage (safe or block) without needing escalation. Track B's fields stay
  null/empty in that case — that's expected, not a bug.
- `explanation` is what the trace visualizer renders directly. Write it as
  if explaining the verdict to someone who hasn't read either paper.
- Extend this schema by adding fields, not renaming existing ones — Track A
  and Track B will be reading/writing this in parallel from Week 1 onward.
