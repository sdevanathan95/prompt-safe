"""Region tagging: split agent history into labeled, non-overlapping spans.

RTBAS attaches labels to regions rather than whole messages, so that one
poisoned item inside an otherwise clean tool response can be redacted without
discarding the response. Regions are rendered with explicit markers before
being shown to the LM-Judge screener, which reports back the ids it found
load-bearing.

Deliberately agent-agnostic: this module takes plain text and tool names, and
never imports the benchmark. Extracting those from a specific framework's
message objects is the caller's job — eval/harness.py already has the
AgentDojo-specific walk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from middleware.screening.labels import BOTTOM, Confidentiality, Integrity, Label

# Tools whose output carries text authored outside the trust boundary. RTBAS's
# AgentDojo rule: a region incorporating textual data from an external source
# is low-integrity. Names are matched as substrings because suites vary the
# prefix (get_/read_/search_) around the same nouns.
EXTERNAL_CONTENT_TOOLS = (
    "email",
    "inbox",
    "message",
    "channel",
    "webpage",
    "website",
    "web",
    "file",
    "document",
    "review",
    "transaction",
    "bill",
    "calendar",
)

# Tools returning data the user would not want republished. Separate from the
# integrity axis on purpose: a bank statement is trustworthy and secret, an
# attacker's email is untrustworthy and public.
PRIVATE_CONTENT_TOOLS = (
    "email",
    "inbox",
    "message",
    "calendar",
    "transaction",
    "balance",
    "statement",
    "account",
    "contact",
    "file",
)

# Splits a tool response into items at top-level YAML-ish list boundaries —
# one inbox message, one transaction, one search hit per region. Responses
# without list structure stay a single region.
_LIST_ITEM_BOUNDARY = re.compile(r"^- ", re.MULTILINE)

# Who authored a region, when the content says so. Labeling by tool alone
# gives every region from one call the same label, which makes intra-response
# redaction impossible: one poisoned email in an inbox cannot be masked while
# the rest stay visible. RTBAS labels a region low-integrity when it carries
# text from an external source, which is a property of the region, not of the
# tool that fetched it.
_AUTHOR_FIELD = re.compile(
    r"^\s*(?:-\s*)?(?:sender|from|author|posted_by|user)\s*:\s*(\S+)",
    re.MULTILINE | re.IGNORECASE,
)

_REGION_PATTERN = re.compile(
    r"<<(REGION_\d+)>>(.*?)<</\1>>",
    re.DOTALL,
)

# Region markers appearing inside region content itself. The content is
# attacker-reachable — it is the injected text we are screening — so text
# shaped like a closing marker would end its region early and let the
# remainder read to the judge as top-level instruction rather than as data.
_MARKER_LIKE = re.compile(r"<</?REGION_\d+>>", re.IGNORECASE)


@dataclass(frozen=True)
class Region:
    id: str
    content: str
    label: Label
    source_tool: str | None = None


def label_for_tool_output(tool_name: str) -> Label:
    """The label a tool's output carries before any join. Unrecognized tools
    default to the lattice bottom, matching RTBAS's rule that unlabeled data
    is most permissive."""
    name = tool_name.lower()
    integrity = (
        Integrity.UNTRUSTED
        if any(keyword in name for keyword in EXTERNAL_CONTENT_TOOLS)
        else Integrity.TRUSTED
    )
    confidentiality = (
        Confidentiality.PRIVATE
        if any(keyword in name for keyword in PRIVATE_CONTENT_TOOLS)
        else Confidentiality.PUBLIC
    )
    return Label(integrity, confidentiality)


def region_author(content: str) -> str | None:
    """The declared author of a region, if the content names one."""
    match = _AUTHOR_FIELD.search(content)
    return match.group(1).strip().strip('"\'') if match else None


def label_for_region(
    tool_name: str,
    content: str,
    trusted_authors: frozenset[str] = frozenset(),
) -> Label:
    """Refine a tool's label using who actually wrote this region.

    With no trusted authors configured this is exactly the tool-level label,
    which is the conservative reading. Configuring them is what lets one
    poisoned message be redacted while its neighbours survive — without it,
    every region from a call shares a label and redaction can only ever mask
    a whole tool response at a time.
    """
    base = label_for_tool_output(tool_name)
    author = region_author(content)
    if author is None or not trusted_authors:
        return base

    domain = author.rpartition("@")[2].lower()
    is_trusted = author.lower() in trusted_authors or domain in trusted_authors
    return Label(
        Integrity.TRUSTED if is_trusted else Integrity.UNTRUSTED,
        base.confidentiality,
    )


def split_content(content: str) -> list[str]:
    """Break one tool response into item-level spans. Non-overlapping and
    order-preserving: concatenating the results reproduces the input."""
    boundaries = [match.start() for match in _LIST_ITEM_BOUNDARY.finditer(content)]
    if not boundaries:
        return [content] if content else []

    # Text before the first list item (a header line, usually) stays with the
    # first item rather than becoming a region of its own.
    starts = [0] + [b for b in boundaries if b > 0]
    spans = [content[start:end] for start, end in zip(starts, starts[1:] + [len(content)])]
    return [span for span in spans if span.strip()]


def build_regions(
    tool_outputs: list[tuple[str, str]],
    start_index: int = 1,
    trusted_authors: frozenset[str] = frozenset(),
) -> list[Region]:
    """Turn (tool_name, content) pairs into labeled regions with stable ids.

    Ids are assigned in traversal order and are the only handle the screener
    gets, so they must stay stable for the lifetime of one screening call.
    """
    regions: list[Region] = []
    next_index = start_index
    for tool_name, content in tool_outputs:
        for span in split_content(content):
            regions.append(
                Region(
                    id=f"REGION_{next_index}",
                    content=span,
                    label=label_for_region(tool_name, span, trusted_authors),
                    source_tool=tool_name,
                )
            )
            next_index += 1
    return regions


def render_tagged(regions: list[Region]) -> str:
    """Render regions with the markers RTBAS shows the judge.

    Labels are deliberately not rendered. The judge is asked which regions the
    next decision depends on — a question about relevance, not about security —
    and showing it the labels invites it to answer the security question
    instead, which is the policy check's job.
    """
    return "\n".join(
        f"<<{region.id}>>{_strip_markers(region.content)}<</{region.id}>>"
        for region in regions
    )


def _strip_markers(content: str) -> str:
    """Neutralize marker-shaped text inside a region so it cannot break out of
    its own delimiters."""
    return _MARKER_LIKE.sub("[marker removed]", content)


def parse_tagged(text: str) -> list[tuple[str, str]]:
    """Recover (region_id, content) pairs from rendered text."""
    return [(match.group(1), match.group(2)) for match in _REGION_PATTERN.finditer(text)]


def dependency_label(regions: list[Region], relevant_ids) -> Label:
    """Join the labels of the regions the screener marked relevant.

    Ids the screener returned that don't correspond to a real region are
    ignored rather than raising: a judge that hallucinates an id has made a
    performance mistake, not a security one, since an unmatched id contributes
    no label and the region it named was never in the history to begin with.
    """
    wanted = set(relevant_ids)
    label = BOTTOM
    for region in regions:
        if region.id in wanted:
            label = label.join(region.label)
    return label


def labels_by_id(regions: list[Region]) -> dict[str, dict]:
    """Per-region labels in the shape middleware/trace/schema.md expects under
    screened_regions.labels."""
    return {region.id: region.label.to_dict() for region in regions}
