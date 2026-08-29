# Causal, explainable security middleware for tool-calling agents

Pluggable middleware that catches indirect prompt injection by testing
whether an agent's action was actually *caused by* the user's request,
instead of classifying tool output text as suspicious.

Full research grounding and architecture rationale: see
`docs/project-brief.md`.

## How it works

An agent's next tool call is checked in stages, each one only running when
the stage before it couldn't resolve the decision:

1. **Screening** — tool output is tagged trusted/untrusted by region. A
   judge model flags which regions actually matter for the current
   decision; irrelevant regions are masked out before the main agent acts
   on them.
2. **Policy check** — a three-way verdict: clearly safe (auto-execute),
   clearly a violation (block), or genuinely ambiguous (escalate).
3. **Counterfactual test** — for ambiguous cases, the agent's step is
   re-run with the user's real task swapped for a neutral placeholder,
   same tool output kept in context. If the real run and the placeholder
   run converge on the same tool call, the action wasn't driven by the
   user's task at all — it was driven by the tool output. That's the
   injection signal. Divergence means the action was genuinely
   task-driven.
4. **Human confirmation** — the rare last resort, only reached if the
   counterfactual test itself is inconclusive.
5. **Trace logging** — every step records what was screened, what was
   masked, and (for any step that reached the counterfactual test) the
   original-vs-masked comparison that produced the verdict. This is what
   makes a block explainable instead of a bare refusal.

## Repo structure

```
middleware/
  screening/   Provenance tagging, judge-based region screener,
               three-way policy check
  melon/       Counterfactual masking engine: masked re-execution,
               tool-call comparison, embedding threshold logic
  trace/       Trace/state schema, logging — the artifact any
               visualizer reads
adapters/      Provider adapters (OpenAI, Anthropic) + framework adapter
               (LangGraph)
eval/          Benchmark harness (AgentDojo / InjecAgent), metrics
               reporting, hand-crafted test scenarios
demo/          Live demo scenario, trace visualizer
docs/          Project brief and design notes
```

## The shared trace contract

Every stage reads/writes the same trace object, defined in
`middleware/trace/schema.md`:

```json
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

## Setup

```
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Tests

```
source .venv/bin/activate
python -m pytest tests/ -v
```
