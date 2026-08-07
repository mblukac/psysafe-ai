"""Categorical attribution shared by user-signal classifiers."""

from enum import Enum


class EvidenceSubject(str, Enum):
    """Whom a user-authored statement concerns."""

    USER = "user"
    THIRD_PARTY = "third_party"
    UNCLEAR = "unclear"


class SourceContext(str, Enum):
    """How evidence is presented within a user-authored message.

    ``fictional`` takes precedence when invented material is also formatted as
    a quotation. ``quoted`` is reserved for a real statement being repeated.
    """

    DIRECT = "direct"
    QUOTED = "quoted"
    FICTIONAL = "fictional"
    UNCLEAR = "unclear"


def is_direct_user_evidence(subject: EvidenceSubject, source_context: SourceContext) -> bool:
    """Return whether attributed evidence is safe for current-user routing.

    Observations retain other subjects and contexts for auditability, but a
    quoted, fictional, third-party, or unclear statement must not become a
    gate-ready decision about the current user.
    """

    return subject is EvidenceSubject.USER and source_context is SourceContext.DIRECT


__all__ = ["EvidenceSubject", "SourceContext", "is_direct_user_evidence"]
