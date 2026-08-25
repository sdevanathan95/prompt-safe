# Causal, explainable security middleware for tool-calling agents

Pluggable middleware that catches indirect prompt injection by testing whether
an agent's action was actually *caused by* the user's request, instead of
classifying tool output text as suspicious.

Full project brief (research grounding, architecture rationale, evaluation
plan): see `docs/project-brief.md`.

## Repo structure

```
middleware/
  screening/   Track A — RTBAS-style provenance tagging, LM-Judge screener,
               3-way policy check (safe / block / escalate)
  melon/       Track B — masked re-execution engine, tool-call comparison,
               embedding threshold logic
  trace/       Shared — trace/state schema, logging, the artifact the
               visualizer reads
adapters/      Provider adapters (OpenAI, Anthropic) + framework adapter
               (LangGraph) — built in Week 3, shared
eval/         Track B — AgentDojo / InjecAgent harness, metrics reporting
demo/         Shared — live demo scenario, trace visualizer
docs/         Project brief and any design notes
```

## The one shared contract

Both tracks read/write the same trace object defined in
`middleware/trace/schema.md`. Agree on this before splitting off — it's the
only thing that has to be right for the two tracks to integrate cleanly in
Week 2.

```
{
  "step": int,
  "source_provenance": "trusted" | "untrusted",
  "screened_regions": [...],
  "policy_verdict": "safe" | "block" | "escalate",
  "melon_check": {
    "ran": bool,
    "original_calls": [...],
    "masked_calls": [...],
    "distance": float,
    "verdict": "safe" | "block" | null
  } | null,
  "final_action": "execute" | "block" | "ask_user"
}
```

## Ownership

| Track | Owns | Weeks |
|---|---|---|
| A | `middleware/screening/`, `middleware/trace/` | 1, then integration in 2–3 |
| B | `middleware/melon/`, `eval/` | 1–2, then integration in 2–3 |
| Shared | `adapters/`, `demo/` | 3 |

See `docs/task-split.md` for the week-by-week breakdown and integration
checkpoints.
