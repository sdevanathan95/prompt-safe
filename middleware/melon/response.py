"""Counterfactual detection on the response channel.

MELON compares tool calls and deliberately not response text (its §3.2
challenge 3: the text is noisy even when the dangerous call is identical). The
cost of that choice is stated in its own failure analysis — 72.73% of the
attacks that evaded it were response-based, where the attacker's goal is met
by what the agent *says* and no tool is called at all. A defense that watches
only tool calls cannot see them, and the attacker only has to stop calling
tools.

Comparing responses naively does not work, for the reason MELON gives. If the
user asks to summarize an inbox and the masking prompt also asks to
summarize, both runs summarize the same content and converge on every benign
case.

Two changes make the comparison sound:

**Task-orthogonal masking.** Pick a masking prompt whose surface task is
maximally unlike the user's — grammar-checking against a summarization
request. A grammar-checking run has no reason to restate the inbox's claims,
so shared content is no longer explained by "both runs did the same job". It
is explained by both runs following an instruction embedded in the content.

**Provenance rather than text similarity.** Compare the distinctive entities
each response asserts that (a) trace back to an untrusted region and (b) do
not appear in the user's own request. Content the user asked about is
excluded by construction, so a legitimate summary contributes nothing to the
comparison. An injected claim contributes to both.

This is a taint-tracking argument applied to the agent's own words, which the
IFC line of work (RTBAS, FIDES, AgentArmor) applies only to tool arguments,
and it needs no trained classifier — the property being tested is where a
string came from, which is checkable.

STATUS: off by default (`check_response_channel`), because measured on the
travel suite it does not yet hold up in either configuration.

  entity overlap alone      3 of 6 benign runs blocked; caught the
                            response-only attack
  overlap + alignment gate  1 of 6 benign runs blocked; caught none of 3

Both are worse than leaving it off, which costs the response-only attacks but
keeps the tool-mediated result at 100% prevention with no false positives. The
failure is informative and specific: entity provenance establishes that
untrusted content reached the response, but not whether the response
*advocates* it rather than merely naming it, and a task about Paris hotels
makes both runs name the same hotels. Separating mention from advocacy is the
open problem here; it is not something the entity level can express.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from middleware.screening.labels import Integrity
from middleware.screening.regions import Region

# Entities short enough to collide by chance carry no evidence. Four
# characters is the same floor the argument-provenance check uses.
MIN_ENTITY_LENGTH = 4

# Capitalised runs and quoted spans are what an assertion is *about* — a hotel
# name, a recipient, a product. Bare prose is not extracted, so ordinary
# phrasing shared between two responses contributes nothing.
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][\w'-]+)(?:\s+(?:[A-Z][\w'-]+|of|the|de|von))*\b")
_QUOTED = re.compile(r"['\"]([^'\"]{4,60})['\"]")
_URL_OR_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b|https?://\S+|\bwww\.\S+")

_NORMALIZE = re.compile(r"[\s\-_(),.]+")

# Sentence-initial capitals and common openers are not entities.
_STOPWORDS = frozenset(
    """the this that these those there here it its a an and or but if then so
    you your yours i me my we our they them he she his her hello hi please
    thanks thank sure certainly here's based according note important system
    summary response answer okay yes no""".split()
)


def _normalized(text: str) -> str:
    return _NORMALIZE.sub("", text).casefold()


def extract_entities(text: str) -> set[str]:
    """The distinctive things a response asserts about."""
    found: set[str] = set()
    for match in _URL_OR_EMAIL.finditer(text):
        found.add(match.group(0))
    for match in _QUOTED.finditer(text):
        found.add(match.group(1))
    for match in _PROPER_NOUN.finditer(text):
        candidate = match.group(0).strip()
        if candidate.casefold() in _STOPWORDS:
            continue
        if len(_normalized(candidate)) >= MIN_ENTITY_LENGTH:
            found.add(candidate)
    return found


def untrusted_assertions(
    text: str,
    regions: list[Region],
    task_description: str,
) -> set[str]:
    """Entities a response asserts that came from untrusted content and that
    the user never mentioned.

    Both conditions matter. Untrusted-but-requested content is what a
    summarization task is *for*, and requiring absence from the task is what
    keeps this from flagging every legitimate summary.
    """
    untrusted = [r for r in regions if r.label.integrity is Integrity.UNTRUSTED]
    if not untrusted:
        return set()

    task = _normalized(task_description)
    assertions = set()
    for entity in extract_entities(text):
        normalized = _normalized(entity)
        if not normalized or normalized in task:
            continue
        if any(normalized in _normalized(region.content) for region in untrusted):
            assertions.add(entity)
    return assertions


@dataclass
class ResponseVerdict:
    converged: bool
    shared: set[str] = field(default_factory=set)
    explanation: str = ""

    def to_trace_dict(self) -> dict:
        return {
            "converged": self.converged,
            "shared_assertions": sorted(self.shared),
            "explanation": self.explanation,
        }


def shared_untrusted_assertions(
    original_response: str,
    masked_response: str,
    regions: list[Region],
    task_description: str,
) -> set[str]:
    """Entities both runs assert that came from untrusted content.

    Necessary but nowhere near sufficient, and this is the hard-won part: an
    entity appearing in both responses does not mean either run was following
    an instruction. A task about Paris hotels makes the agent name hotels, and
    a masked run over hotel reviews names the same hotels, because that is
    what the content is about. Measured directly: used alone as the verdict,
    this flagged 3 of 6 benign travel runs on names like "Montmartre Suites".

    So this is only the cheap gate. Whether the response *advocates* those
    entities rather than merely mentioning them is a question about intent
    against the user's request, and is answered by the alignment judge.
    """
    original = untrusted_assertions(original_response, regions, task_description)
    if not original:
        return set()
    masked = untrusted_assertions(masked_response, regions, task_description)
    return {
        entity
        for entity in original
        if any(_normalized(entity) == _normalized(other) for other in masked)
    }


def compare_responses(
    original_response: str,
    masked_response: str,
    regions: list[Region],
    task_description: str,
    serves_user_task=None,
) -> ResponseVerdict:
    """Is the response carrying an injected assertion?

    Two conditions, and both are needed. The entity overlap establishes that
    untrusted content survived into a run that never had the user's task. The
    alignment check establishes that asserting it is not what the user asked
    for — which is what separates an injected recommendation from an ordinary
    answer that happens to name something it read.

    `serves_user_task` is a callable returning True when the response does
    serve the request; None disables the second condition, in which case
    nothing is flagged rather than everything.
    """
    shared = shared_untrusted_assertions(
        original_response, masked_response, regions, task_description
    )
    if not shared:
        return ResponseVerdict(
            False,
            set(),
            "The response asserts nothing from untrusted content that also "
            "survived removing the user's task.",
        )

    if serves_user_task is None:
        return ResponseVerdict(
            False,
            shared,
            "The response repeats untrusted content, but without a check of "
            "whether the user asked for it there is no basis to call it an "
            "injection.",
        )

    if serves_user_task():
        return ResponseVerdict(
            False,
            shared,
            "The response draws on content the agent read, but that is what "
            "the user's request called for.",
        )

    listed = ", ".join(sorted(shared)[:3])
    return ResponseVerdict(
        True,
        shared,
        f"The response asserts {listed}, which came from content the agent "
        f"read rather than from the user, and which a run with the user's task "
        f"removed asserted too. Nothing in the request asked for it, so the "
        f"content is what prompted it — an injection carried by the response "
        f"rather than by a tool call.",
    )


# Margin on the similarity difference above which the real run is judged to
# have been following embedded instructions rather than describing content.
# It applies to a *difference* of two cosines measured under otherwise
# identical conditions, not to a similarity, so it does not have to absorb
# topic overlap, verbosity or register — those appear in both terms and
# cancel. That is the whole reason this quantity is thresholdable at all when
# the raw similarity was not.
DEFAULT_COMPLIANCE_MARGIN = 0.05


@dataclass
class DifferentialVerdict:
    complied: bool
    delta: float
    similarity_to_follower: float
    similarity_to_describer: float
    explanation: str = ""

    def to_trace_dict(self) -> dict:
        return {
            "complied": self.complied,
            "delta": round(self.delta, 4),
            "similarity_to_follower": round(self.similarity_to_follower, 4),
            "similarity_to_describer": round(self.similarity_to_describer, 4),
            "explanation": self.explanation,
        }


def differential_convergence(
    original_response: str,
    follower_response: str,
    describer_response: str,
    margin: float = DEFAULT_COMPLIANCE_MARGIN,
) -> DifferentialVerdict:
    """Did the real run behave like an instruction-follower or a describer?

    Two masked runs over the same content, differing only in whether they were
    told to obey instructions found in it. Whichever the real response
    resembles more says which behaviour it exhibited — without ever needing to
    decide, in the abstract, how similar is "too similar".
    """
    from adapters.embeddings import cosine_similarity, embed_many

    if not (original_response and follower_response and describer_response):
        return DifferentialVerdict(
            False, 0.0, 0.0, 0.0,
            "One of the three runs produced no text, so there is nothing to "
            "compare.",
        )

    original, follower, describer = embed_many(
        [original_response, follower_response, describer_response]
    )
    to_follower = cosine_similarity(original, follower)
    to_describer = cosine_similarity(original, describer)
    delta = to_follower - to_describer

    if delta <= margin:
        return DifferentialVerdict(
            False, delta, to_follower, to_describer,
            f"The response resembles a run told to ignore embedded "
            f"instructions about as much as one told to obey them "
            f"(difference {delta:+.3f}). It described what it read rather "
            f"than acting on it.",
        )

    return DifferentialVerdict(
        True, delta, to_follower, to_describer,
        f"With the user's task removed, a run told to obey instructions found "
        f"in the content produced something markedly closer to this response "
        f"than a run told to ignore them did (difference {delta:+.3f}). The "
        f"response is carrying out an instruction that came from the content, "
        f"not from the user.",
    )
