"""Render trace files as a self-contained HTML report.

The point of the whole pipeline is that a block is explainable rather than a
bare refusal, so this shows the path a verdict actually took: which regions
were seen, what each was labeled, which ones the screener found load-bearing,
the label comparison that produced the policy verdict, and — for steps that
escalated — the original-versus-masked tool calls that settled it.

Usage:
    python -m demo.visualize traces.jsonl -o report.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

_VERDICT_CLASS = {"safe": "ok", "execute": "ok", "escalate": "warn", "block": "bad", "ask_user": "warn"}

_CSS = """
:root { --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b66; --line:#e2e2dd;
        --ok:#2f6f4f; --warn:#8a6420; --bad:#9b3232; --card:#fff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161614; --fg:#eceae4; --muted:#9a978e; --line:#33322d;
          --ok:#7fc4a0; --warn:#d8ab5e; --bad:#e08585; --card:#1e1d1a; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:60rem; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1rem; margin:2.5rem 0 .75rem; }
.sub { color:var(--muted); margin:0 0 2rem; }
.case { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1.1rem 1.25rem; margin-bottom:1rem; }
.case-head { display:flex; flex-wrap:wrap; gap:.5rem; align-items:baseline;
             justify-content:space-between; margin-bottom:.75rem; }
.case-id { font-weight:600; }
.tag { font-size:.75rem; padding:.15rem .5rem; border-radius:999px;
       border:1px solid currentColor; white-space:nowrap; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem;
        font-size:.85rem; margin:.6rem 0 .9rem; }
.flow span.step { border:1px solid var(--line); border-radius:6px; padding:.2rem .5rem; }
.arrow { color:var(--muted); }
.explain { font-size:.9rem; color:var(--fg); background:transparent;
           border-left:3px solid var(--line); padding:.4rem 0 .4rem .8rem; margin:.5rem 0; }
table { border-collapse:collapse; width:100%; font-size:.82rem; margin:.5rem 0; }
th,td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:500; }
code { font:.82em ui-monospace,SFMono-Regular,Menlo,monospace; }
.regions { display:flex; flex-wrap:wrap; gap:.3rem; margin:.4rem 0; }
.r { font:.75rem ui-monospace,monospace; padding:.15rem .45rem; border-radius:5px;
     border:1px solid var(--line); }
.r.rel { border-color:var(--ok); color:var(--ok); }
.r.msk { opacity:.5; text-decoration:line-through; }
.scroll { overflow-x:auto; }
.legend { font-size:.8rem; color:var(--muted); margin-top:.4rem; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _label(label: dict | None) -> str:
    if not label:
        return "—"
    return f"{label['integrity']}, {label['confidentiality']}"


def _tag(value: str) -> str:
    return f'<span class="tag {_VERDICT_CLASS.get(value, "")}">{_esc(value)}</span>'


def _regions_block(screened: dict) -> str:
    relevant = set(screened.get("relevant", []))
    masked = set(screened.get("masked", []))
    labels = screened.get("labels", {})
    if not labels:
        return '<p class="legend">No regions recorded for this step.</p>'

    chips = []
    for region_id, label in labels.items():
        css = "rel" if region_id in relevant else ("msk" if region_id in masked else "")
        title = f"{region_id}: {_label(label)}"
        chips.append(f'<span class="r {css}" title="{_esc(title)}">{_esc(region_id)}</span>')
    return (
        f'<div class="regions">{"".join(chips)}</div>'
        f'<p class="legend">{len(labels)} regions · '
        f'{len(relevant)} load-bearing · {len(masked)} redacted. '
        f'Hover a region for its label.</p>'
    )


def _melon_block(melon: dict | None) -> str:
    if not melon or not melon.get("ran"):
        return (
            '<p class="legend">The counterfactual test did not run — the policy '
            "check settled this step on its own.</p>"
        )
    rows = []
    for title, calls in (("original run", melon.get("original_calls", [])),
                         ("masked run", melon.get("masked_calls", []))):
        rendered = ", ".join(
            f"<code>{_esc(c['name'])}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})</code>"
            for c in calls
        ) or '<span class="legend">no tool calls</span>'
        rows.append(f"<tr><th>{title}</th><td>{rendered}</td></tr>")
    distance = melon.get("distance")
    distance_text = f"{distance:.3f}" if isinstance(distance, (int, float)) else "—"
    rows.append(f"<tr><th>distance</th><td><code>{distance_text}</code> "
                f"(converged ⇒ caused by the tool output, not the task)</td></tr>")
    return f'<div class="scroll"><table>{"".join(rows)}</table></div>'


def render(traces: list[dict]) -> str:
    cases = []
    for trace in traces:
        case_id = trace.get("case", f"step {trace.get('step')}")
        injection = trace.get("injection")
        condition = f"attacked · {injection}" if injection else "benign"

        flow = " ".join(
            f'<span class="step">{part}</span><span class="arrow">→</span>'
            for part in ("regions tagged", "screened", "redacted")
        )
        cases.append(f"""
<section class="case">
  <div class="case-head">
    <span class="case-id">{_esc(case_id)} <span class="legend">({_esc(condition)})</span></span>
    <span>{_tag(trace.get("policy_verdict", "?"))} {_tag(trace.get("final_action", "?"))}</span>
  </div>
  <div class="flow">{flow}<span class="step">policy</span><span class="arrow">→</span>
    <span class="step">{_esc(trace.get("policy_verdict", "?"))}</span></div>
  {_regions_block(trace.get("screened_regions", {}))}
  <div class="scroll"><table>
    <tr><th>data this step depends on</th><td><code>{_esc(_label(trace.get("context_label")))}</code></td></tr>
    <tr><th>what the tool call allows</th><td><code>{_esc(_label(trace.get("policy_label")))}</code></td></tr>
  </table></div>
  {_melon_block(trace.get("melon_check"))}
  <p class="explain">{_esc(trace.get("explanation", ""))}</p>
</section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Injection defense trace</title><style>{_CSS}</style></head>
<body><main>
<h1>Injection defense trace</h1>
<p class="sub">{len(traces)} steps. Each shows why the verdict was reached, not just what it was.</p>
{"".join(cases)}
</main></body></html>"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render a trace file as HTML.")
    parser.add_argument("traces", help="JSON Lines trace file from eval.harness --trace-out")
    parser.add_argument("-o", "--out", default="trace-report.html")
    args = parser.parse_args()

    traces = [json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()]
    Path(args.out).write_text(render(traces), encoding="utf-8")
    print(f"wrote {args.out} ({len(traces)} steps)")


if __name__ == "__main__":
    main()
