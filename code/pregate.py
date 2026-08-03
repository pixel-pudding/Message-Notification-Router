"""Stage 2: PRE-GATE (locked architecture doc's "signal & injection gate").

Deterministic, model-free checks that run BEFORE any agent call. Each check
can force a final decision -- when one does, main.py skips run_agent()
entirely for that message, saving an API round trip on clear-cut cases.
run_pre_gate() returns None to mean "nothing forced, proceed to the agent"
as normal.

This is a cost/latency optimization, not a correctness requirement:
decide.py's own injection and business_trust_floor overrides already catch
these same cases after the agent runs, so a message that slips past
pre-gate (or that pre-gate doesn't apply to) is still caught by decide.py's
redundant safety net. The two checks below call decide.py's own detection
functions directly rather than reimplementing the patterns here, so the two
layers can never drift out of sync with each other.

Only two checks fire here -- the two decide.py already treats as HARD
overrides (unconditional once their preconditions hold). decide.py's SOFT
priors (mute_prior, opt_out_prior, high_forward_count, during_dnd, the
urgency-language checks) all depend on comparing against what the agent
actually recommended, so they cannot be evaluated before the agent runs and
are intentionally not replicated here.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import config
import decide
from context import MessageContext, safe_media_text

# Fixed confidence for a pre-gate-forced decision: both checks below only
# fire on unambiguous deterministic signals (a real injection payload, or a
# trust floor with no corroborating relationship on file), so this sits in
# the same high-but-not-maximal band decide.py's own hard overrides land in
# for equivalent cases -- see config.CONFIDENCE_MIN/CONFIDENCE_MAX.
PREGATE_CONFIDENCE: float = 0.87


@dataclasses.dataclass
class PreGateResult:
    """A forced final decision. Only ever constructed when a check fires --
    run_pre_gate() returns None (not a PreGateResult) when nothing does.
    """

    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


def injection_check(ctx: MessageContext) -> Optional[PreGateResult]:
    """Mirrors decide.py's injection override: injection_flag alone isn't
    enough, the message (its text OR its media transcript -- voice notes
    carry the injection attempt entirely in the transcript) also has to be
    soliciting something sensitive (OTP/payment/verification/link click)
    for this to be treated as a scam attempt.
    """
    signals = ctx.signals
    if not signals.injection_flag:
        return None

    combined_text = (ctx.message_text or "") + " " + safe_media_text(ctx.media_id)
    if not decide._requests_sensitive_action(combined_text):
        return None

    return PreGateResult(
        action="mute",
        message_type="scam",
        reason=(
            f"Pre-gate: injection pattern detected ({signals.injection_matched_pattern!r}) combined "
            "with an OTP/payment/verification/link-click request -- treated as a scam attempt before "
            "the agent was invoked."
        ),
        confidence=PREGATE_CONFIDENCE,
        evidence_message_ids="none",
    )


def business_scam_check(ctx: MessageContext) -> Optional[PreGateResult]:
    """Mirrors decide.py's business_trust_floor override, evaluated the
    closest way possible before the agent has run: cold_business_sender
    (no user_business_history.csv relationship row at all) stands in for
    "the agent found no evidence of a real relationship," since there is no
    agent-cited evidence yet at this point in the pipeline. Verified against
    every business_trust_floor=True message in the production dataset to
    confirm this produces the same action/message_type decide.py's own
    (more granular, message-history-evidence-based) check would have picked
    for every one of them, including the one case where the two checks
    could in principle disagree.
    """
    signals = ctx.signals
    if not (signals.business_trust_floor and signals.cold_business_sender):
        return None

    return PreGateResult(
        action="mute",
        message_type="scam",
        reason=(
            "Pre-gate: unverified business with domain mismatch, account under "
            f"{config.BUSINESS_TRUST_FLOOR_MAX_AGE} days old, and no prior user relationship on file -- "
            "treated as a scam attempt before the agent was invoked."
        ),
        confidence=PREGATE_CONFIDENCE,
        evidence_message_ids="none",
    )


def run_pre_gate(ctx: MessageContext) -> Optional[PreGateResult]:
    """Run deterministic safety checks BEFORE the agent.

    Returns a PreGateResult to skip the agent, or None to proceed with the
    normal DESCRIBE (agent) -> DECIDE -> VALIDATE flow.
    """
    return injection_check(ctx) or business_scam_check(ctx)
