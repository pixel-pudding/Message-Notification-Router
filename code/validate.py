"""Stage 6: VALIDATE (locked architecture doc section 7).

Enforces the exact output schema and allowed values required for output.csv
before a FinalDecision is written to output.csv, repairing whatever it can
rather than raising. No retry logic lives here -- re-running the agent on a
bad result is main.py's job once that orchestration is wired up; this module
only ever looks at the FinalDecision it's given.
"""
from __future__ import annotations

import dataclasses
import logging

import config
from decide import FinalDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate")

DEFAULT_REASON = "No reasoning provided"


@dataclasses.dataclass
class ValidatedOutput:
    """Schema-clean row, ready to write to output.csv as-is."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


def validate_output(decision: FinalDecision, evidence_shortlist: list[dict]) -> ValidatedOutput:
    """Repair `decision` into a ValidatedOutput that's guaranteed to satisfy
    the output schema, logging every repair made:

    1. action must be in config.ALLOWED_ACTIONS, else "digest".
    2. message_type must be in config.ALLOWED_MESSAGE_TYPES, else "unknown".
    3. every id in evidence_message_ids must exist in evidence_shortlist for
       this message; ids that don't are dropped (and logged); "none" if
       nothing survives.
    4. confidence must be a float in [0, 1]; non-numeric -> 0.5, else clipped.
    5. reason must be non-empty, else DEFAULT_REASON.
    """
    action = decision.action
    if action not in config.ALLOWED_ACTIONS:
        logger.warning("message_id=%s: invalid action %r, repairing to 'digest'", decision.message_id, action)
        action = "digest"

    message_type = decision.message_type
    if message_type not in config.ALLOWED_MESSAGE_TYPES:
        logger.warning("message_id=%s: invalid message_type %r, repairing to 'unknown'", decision.message_id, message_type)
        message_type = "unknown"

    evidence_message_ids = _validate_evidence_ids(decision.message_id, decision.evidence_message_ids, evidence_shortlist)

    confidence = decision.confidence
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        logger.warning("message_id=%s: non-numeric confidence %r, repairing to 0.5", decision.message_id, confidence)
        confidence = 0.5
    clipped = max(0.0, min(1.0, confidence))
    if clipped != confidence:
        logger.warning("message_id=%s: confidence %r out of [0, 1], clipping to %r", decision.message_id, confidence, clipped)
    confidence = clipped

    reason = decision.reason
    if not reason or not reason.strip():
        logger.warning("message_id=%s: empty reason, repairing to default", decision.message_id)
        reason = DEFAULT_REASON

    return ValidatedOutput(
        message_id=decision.message_id,
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=evidence_message_ids,
    )


def _validate_evidence_ids(message_id: str, evidence_message_ids: str, evidence_shortlist: list[dict]) -> str:
    if not evidence_message_ids or evidence_message_ids == "none":
        return "none"

    valid_ids = {e["message_id"] for e in evidence_shortlist}
    cited = [eid for eid in evidence_message_ids.split(";") if eid]
    kept = [eid for eid in cited if eid in valid_ids]
    dropped = [eid for eid in cited if eid not in valid_ids]

    if dropped:
        logger.warning("message_id=%s: dropping evidence id(s) not in evidence_shortlist: %s", message_id, dropped)

    return ";".join(kept) if kept else "none"
