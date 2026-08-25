# Task split — 3 weeks, 2 contributors

## Day 0 (do this together, before splitting off)
- Agree on the trace/state schema in `middleware/trace/schema.md` — this is
  the only interface the two tracks share.
- Pick the base agent to wrap (plain ReAct loop is easiest to instrument;
  LangGraph is fine too — just pick one and don't relitigate it later).
- Both read AgentArmor (arXiv:2508.01249) fully before Week 3 — it's the
  least-explored of the three papers and Stage 5's trace format depends on it.

## Week 1

**Track A (screening owner)**
- Provenance tagging on tool outputs (trusted/untrusted).
- Region tagging + LM-Judge screener (prompt the judge to reason over
  `«REGION_N»`-tagged history, return which regions are load-bearing).
- Masking: redact irrelevant regions from what the main agent LM sees.
- 3-way policy check: safe / block / escalate.
- Trace logging skeleton — write every step's verdict to the shared schema
  even before Track B's fields are populated.
- Milestone: on 2–3 hand-crafted injection scenarios, the screener correctly
  routes to "escalate" (Track B not built yet, so escalate = TODO stub).

**Track B (causal test owner)**
- Masked re-execution engine: neutral placeholder task substitution (not
  blank masking — this is the fix for MELON's challenge ❶).
- Tool-call cache to keep original/masked runs synchronized on content
  rather than step index (fix for challenge ❷).
- Comparison logic on tool calls only, not full text+calls (fix for
  challenge ❸) — embed with a text embedding model, threshold the distance.
- Milestone: given a hand-fed (original_action, masked_action) pair, the
  engine returns a verdict + distance score. No integration with Track A yet
  — test it standalone against scripted scenarios.

## Week 2

**Track A**
- Widen attack coverage beyond the single email scenario — add 2–3 more
  scenario types (banking, travel, from AgentDojo's domains).
- Integrate with Track B: when policy check returns "escalate", call into
  `middleware/melon/` with the current state, get back safe/block.
- Start populating the shared trace schema's `melon_check` fields from
  Track B's output.

**Track B**
- Wire the re-execution engine to actually run against the live agent (not
  just scripted pairs) — this means it needs to call the real LLM twice.
- Cost-control heuristic: a cheap pre-filter so Stage 3 doesn't fire on
  every escalate if it's obviously unnecessary — worth a day here, it's also
  a legitimate thing to mention as a contribution in the writeup.
- Start the AgentDojo/InjecAgent eval harness — get one real number (however
  rough) on attack prevention + false positive rate by end of week.

**Checkpoint (end of Week 2):** full pipeline runs end to end on at least
one real benchmark scenario — screening → escalate → MELON test → verdict —
with the trace schema fully populated at every stage.

## Week 3

**Shared**
- Provider adapters: OpenAI + Anthropic tool-calling schema translation for
  the middleware's own internal calls (judge call, masked re-execution call).
- LangGraph adapter — the one-line integration for the demo.
- Trace visualizer (reads the shared schema, renders the data-flow graph:
  tool output → reasoning step → tool call → masked-run comparison if any).
- Full benchmark run: report attack prevention rate, false-positive rate,
  latency overhead by stage, and the headline number — reduction in
  human-confirmation rate vs. RTBAS's original binary fallback, plus
  accuracy of the automated resolutions that replaced those confirmations.
- Demo polish: the email-forwarding scenario, live, against two different
  LLM providers through the same middleware, trace rendered for both.

## What NOT to do
- Don't build the attention-based screener (RTBAS's other option) — it
  needs training data/infra neither of you has time for. LM-Judge only.
- Don't build a full network proxy. A decorator around the tool-execution
  function is enough and is what the demo needs.
- Don't chase MCP-specific attacks unless Weeks 1–2 finish early — it's a
  stretch goal, not core to the pitch.
