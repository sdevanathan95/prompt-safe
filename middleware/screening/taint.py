"""Taint that survives a write, so it cannot be laundered through the
environment.

Labels are computed from the transcript: `regions.build_regions` decides
integrity from the tool that produced a span and the author named inside it.
That is sound for content the agent reads directly and blind to a round trip
through the environment:

    step 1  read_email()          -> poisoned text, UNTRUSTED
    step 2  create_note(body=...) -> the agent copies it into the user's notes
    step 5  read_notes()          -> authored by the user's own app, TRUSTED

The write launders the label. Nothing in the transcript notices, because after
step 2 the author genuinely is the user. AgentDojo barely exercises write-then-
read round trips, which is exactly what makes this dangerous: it does not show
up in any benchmark number reported here, while real agents with scratchpads,
notes or memory do it constantly. The 2025-2026 memory-poisoning literature
makes the same point structurally -- defenses built for prompt injection do not
cover persistence, because the payload need carry no detectable pattern at the
point it is read back.

**Taint follows values, not object identity.** The obvious design keys a store
on the written object (file path, note id, event id) and recovers its label on
a later read. That needs an answer to "is this the same object after an edit?"
and "does a whole file inherit the label of one appended line?", neither of
which has a good one. Tracking the distinctive *values* that crossed the
boundary sidesteps both: an attacker's payload has to survive the round trip
to be useful, and if it survived, it is present to be recognised. Rephrasing it
on the way out breaks the attack rather than the detector, because what gets
read back is then no longer the attacker's instruction.

This is deliberately not an LLM call. The question is whether a literal value
appears in a literal span, which is checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

from middleware.screening.labels import Integrity, Label
from middleware.screening.provenance import (
    _contains,
    argument_label,
    is_distinctive,
)
from middleware.screening.regions import Region


@dataclass(frozen=True)
class TaintedValue:
    value: str
    label: Label
    written_by: str


class TaintStore:
    """Values that reached the environment carrying a restrictive label.

    One store per session. It only ever grows: a value that was untrusted when
    written does not become trustworthy because it was copied again, and the
    number of distinctive values one agent turn writes is small.
    """

    def __init__(self) -> None:
        self._tainted: list[TaintedValue] = []

    def __len__(self) -> int:
        return len(self._tainted)

    @property
    def values(self) -> list[TaintedValue]:
        return list(self._tainted)

    def record_write(
        self,
        tool_name: str,
        arguments: dict,
        regions: list[Region],
        task_description: str,
        fallback: Label,
    ) -> None:
        """Remember the values this call carried out of untrusted content.

        Per argument, not per call. Recording every argument of an untrusted
        call taints values that merely travelled beside the payload -- a
        `title="todo"` written alongside a poisoned body would then pull down
        every future region containing the word "todo". Only values whose own
        provenance is untrusted are worth recovering, and `argument_label`
        already answers that.
        """
        for value in arguments.values():
            if not is_distinctive(value):
                continue
            label = argument_label(value, regions, task_description, fallback)
            if label.integrity is not Integrity.UNTRUSTED:
                continue
            text = str(value)
            if not any(t.value == text for t in self._tainted):
                self._tainted.append(TaintedValue(text, label, tool_name))

    def relabel(self, regions: list[Region]) -> list[Region]:
        """Restore the label of any region carrying a value written earlier.

        Joins rather than overwrites, so a region that is already private stays
        private and only its integrity is pulled down.
        """
        if not self._tainted:
            return regions

        restored: list[Region] = []
        for region in regions:
            carried = [t for t in self._tainted if _contains(region.content, t.value)]
            if not carried:
                restored.append(region)
                continue
            label = region.label
            for tainted in carried:
                label = label.join(tainted.label)
            restored.append(
                Region(
                    id=region.id,
                    content=region.content,
                    label=label,
                    source_tool=region.source_tool,
                    source_arguments=region.source_arguments,
                )
            )
        return restored

    def explain(self, region: Region) -> str | None:
        """Why a region was pulled back down, for the trace. None when it was
        not."""
        carried = [t for t in self._tainted if _contains(region.content, t.value)]
        if not carried:
            return None
        source = carried[0].written_by
        return (
            f"{region.id} reads back a value this session wrote out through "
            f"{source}, and that value came from untrusted content. The write "
            "does not make it trustworthy."
        )
