# Shared trace schema

This is the one contract both tracks must agree on before splitting work.
Every step of the agent's execution produces one of these objects.

```json
{
  "step": 0,
  "source_provenance": "trusted | untrusted",
  "screened_regions": {
    "relevant": ["region_ids flagged load-bearing by the LM-Judge"],
    "masked": ["region_ids redacted from the agent's context this step"]
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
- `melon_check.ran` is `false` for any step that resolved at the policy-check
  stage (safe or block) without needing escalation. Track B's fields stay
  null/empty in that case — that's expected, not a bug.
- `explanation` is what the trace visualizer renders directly. Write it as
  if explaining the verdict to someone who hasn't read either paper.
- Extend this schema by adding fields, not renaming existing ones — Track A
  and Track B will be reading/writing this in parallel from Week 1 onward.
