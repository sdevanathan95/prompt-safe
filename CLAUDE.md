# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project is

Causal, explainable security middleware for tool-calling LLM agents. Catches
indirect prompt injection by testing whether an agent's action was actually
*caused by* the user's task, not by classifying tool-output text as
suspicious. See `docs/project-brief.md` for the full research grounding
(MELON, RTBAS, AgentArmor) and `TRACK_B_PLAN.md` for the current
implementation plan.

Two tracks, one shared contract (`middleware/trace/schema.md`):
- Track A: `middleware/screening/`, `middleware/trace/` — RTBAS-style
  provenance tagging, LM-Judge region screener, 3-way policy check.
- Track B: `middleware/melon/`, `eval/` — MELON counterfactual masking,
  tool-call comparison, embedding threshold.

## Environment

- Python, managed with `uv`. Venv at `.venv/`, deps in `requirements.txt`.
- Run tests: `source .venv/bin/activate && python -m pytest tests/ -v`.
- New dependencies go in `requirements.txt`, pin as you go.

## Coding standard: write this like a distinguished engineer would

That means, concretely:

- **Correctness over cleverness.** Prefer the boring, obviously-correct
  implementation. If a clever trick saves 3 lines but costs 30 seconds of
  reader comprehension, don't.
- **No speculative generality.** Don't build config systems, plugin
  registries, or abstract base classes for a single concrete need. Three
  similar call sites beat one premature abstraction. Add the abstraction
  when the third real use case shows up, not before.
- **Every design decision traces to a reason.** This codebase implements
  specific fixes for specific documented failure modes (MELON's three
  engineering challenges, RTBAS's label-creep problem). When you write code
  that addresses one of these, say which one in a comment — one line, not a
  restatement of the paper. When you make a judgment call the papers don't
  settle (threshold values, alignment strategy, cache eviction), say so
  explicitly rather than silently picking a number.
- **Types and interfaces are contracts, not decoration.** The shared trace
  schema (`middleware/trace/schema.md`) is the one thing both tracks depend
  on — never rename its fields, only add to it. Within a track, dataclasses
  and function signatures should make illegal states hard to represent, not
  just document intent in a docstring.
- **Fail loud on ambiguity, fail quiet on expected absence.** A missing
  masked-run counterpart for a tool call is expected and meaningful (max
  divergence) — handle it as data, not an exception. A malformed trace
  object or a threshold that's `None` when it shouldn't be is a bug —
  raise, don't silently coerce.
- **No dead scaffolding.** Don't leave TODO stubs, commented-out
  alternative implementations, or unused parameters "for future
  flexibility." If it's not load-bearing now, delete it; git history is the
  changelog.
- **Test the failure mode, not just the happy path.** Every comparator or
  policy-check change should come with at least one scenario where the
  naive implementation would have gotten it wrong (e.g.
  `injection_same_tool_different_recipient` in
  `eval/scenarios/hand_crafted.py` exists specifically to catch a
  comparator that matches on tool name alone).
- **Numbers that matter get named, not inlined.** Thresholds, model names,
  distance metrics — module-level constants with a comment on provenance
  (paper default vs. placeholder vs. tuned-against-data), never magic
  literals buried in a function body.
- **Explain verdicts like a colleague, not a log line.** Anything that
  produces a `verdict`/`explanation` pair (policy check, MELON compare)
  should write the explanation for someone who hasn't read the papers —
  that's the whole point of the explainability pitch in
  `docs/project-brief.md` §7.

## Comments

- Short. One line, not a paragraph.
- Neutral and self-contained: describe the code's own logic (why this
  branch exists, what invariant it protects), never the surrounding task,
  request, or conversation that produced it. No "fix for", "per the user's
  request", "added because", "handles the case from issue #123".
- If a comment would only make sense to someone who read the current
  conversation, it doesn't belong in the file — drop it or rewrite it so it
  stands alone.
- Skip comments a well-named function/variable already makes obvious.

## Testing

- Every new function or changed behavior gets a unit test in `tests/`,
  written in the same change that introduces it — not deferred.
- Cover the failure mode, not just the happy path (see
  `injection_same_tool_different_recipient` in
  `eval/scenarios/hand_crafted.py` for the pattern: a scenario that exists
  specifically to catch a plausible-but-wrong implementation).
- Run `python -m pytest tests/ -v` before considering any change complete.

## README.md

- Update `README.md` in the same change as any code change that alters
  repo structure, setup steps, or what a component does — not as a
  follow-up. Treat it as out of sync, not optional, if left behind.
- `README.md` describes the application only: what it does, how it's
  architected, how to install/run/test it. It is user- and
  contributor-facing documentation of the software itself.
- Keep out anything about how the work is organized or split: no track/
  owner assignments, no week-by-week plans, no contributor names, no
  effort/scheduling breakdowns. That belongs in planning docs (e.g.
  `TRACK_B_PLAN.md`, `docs/task-split.md`), never in `README.md`.

## Before calling something done

- Tests pass (`python -m pytest tests/ -v`).
- New scenarios added for any new comparator/policy logic, not just the
  case that was easy to construct.
- `README.md` reflects the current state of the repo.
- If you touched `middleware/trace/schema.md`'s shape, both tracks'
  consumers still parse it — check, don't assume.
