"""Selective masking: remove from the agent's view anything more restrictive
than what its next decision actually depends on.

The rule is a label comparison, not a set difference. A region is kept when its
own label flows to the dependency label the screener computed; otherwise it is
replaced with a redaction marker. An irrelevant untrusted region disappears as
a *consequence* of its label not having been joined into the dependency label —
not because the screener named it for removal.

Getting this backwards (deleting exactly what the screener called irrelevant)
looks equivalent and is not: it would also delete irrelevant regions that are
harmless, starving the agent of context, while keeping relevant-but-restrictive
ones the label comparison would have removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from middleware.screening.labels import Label
from middleware.screening.regions import Region

# RTBAS writes the redacted stand-in as ◊. Kept verbatim so a trace read
# alongside the paper is unambiguous.
REDACTION_MARKER = "◊"


@dataclass
class RedactionResult:
    kept: list[Region]
    masked_ids: list[str]
    text: str


def is_visible(region: Region, dependency_label: Label) -> bool:
    """A region survives iff its own label flows to the dependency label."""
    return region.label.leq(dependency_label)


def masked_region_ids(regions: list[Region], dependency_label: Label) -> list[str]:
    return [r.id for r in regions if not is_visible(r, dependency_label)]


def redact(regions: list[Region], dependency_label: Label) -> RedactionResult:
    """Produce what the main agent LM is allowed to see this step."""
    kept: list[Region] = []
    masked_ids: list[str] = []
    lines: list[str] = []

    for region in regions:
        if is_visible(region, dependency_label):
            kept.append(region)
            lines.append(region.content)
        else:
            masked_ids.append(region.id)
            lines.append(REDACTION_MARKER)

    return RedactionResult(kept=kept, masked_ids=masked_ids, text="\n".join(lines))
