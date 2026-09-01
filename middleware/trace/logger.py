"""Append step traces to a JSON Lines file.

One object per line so a run can be tailed while it is still going and so a
truncated run is still parseable up to its last complete step — which matters
because a benchmark run that dies partway through is otherwise unreadable.
"""

from __future__ import annotations

import json
from pathlib import Path

from middleware.trace.schema import StepTrace


class TraceLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.steps: list[StepTrace] = []

    def log(self, trace: StepTrace) -> None:
        self.steps.append(trace)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict()) + "\n")


def read_traces(path: str | Path) -> list[dict]:
    """Read back a trace file, skipping a trailing partial line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    traces = []
    for line in lines:
        if not line.strip():
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return traces
