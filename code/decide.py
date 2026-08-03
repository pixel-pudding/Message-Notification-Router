"""Stage 5: DECIDE (locked architecture doc sections 3, 5, 6).

Combines the agent's structured AgentOutput (or None if the agent failed)
with the deterministic hard signals already attached to MessageContext into
the final action/message_type/reason/confidence/evidence_message_ids. This
is the only stage allowed to produce final decision fields -- the model
never does, and nothing here calls the model.

NOTE: named decide.py (not decision.py) to match the file already scaffolded
for this pipeline stage in code/main.py's imports; same "DECIDE" role either
way.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from typing import Optional

import config
from agent import AgentOutput
from context import MessageContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("decide")

FALLBACK_REASON = "Fallback: agent could not produce a valid understanding"
FALLBACK_CONFIDENCE = 0.5  # deliberately below CONFIDENCE_MIN -- signals "not confident" for a flagged-for-review row

# Architecture §3 injection override: injection_flag alone isn't enough --
# it also requires the message to be soliciting something sensitive.
SENSITIVE_ACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\botp\b",
        r"one[- ]time password",
        r"\bpin\b",
        r"verify|verification",
        r"\bpassword\b",
        r"\bpayment\b|\bpay\b",
        r"click (here|below|the link)",
        r"tap (here|below)",
        r"\blink\b",
    ]
]

UNCERTAINTY_MARKERS = ("uncertain", "unclear", "might", "possibly")

# Eval finding (30-sample baseline): the agent's own reasoning sometimes
# already contains the correct urgency read even when recommended_action
# doesn't reflect it (e.g. reasoning says "Nothing urgent" but still
# recommends notify) -- these two lists let make_decision cross-check the
# recommendation against what the agent actually described.
LOW_URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"not urgent",
        r"non-urgent",
        r"nothing urgent",
        r"no urgency",
        r"isn't urgent",
        r"not an emergency",
        r"no immediate action",
        r"informational",
        r"no rush",
        r"can wait",
        r"low priority",
        r"\bFYI\b",
        r"for your reference",
    ]
]

HIGH_URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\burgent\b",
        r"immediately",
        r"\basap\b",
        r"right away",
        r"time[- ]sensitive",
        r"emergency",
        r"deadline",
        r"as soon as possible",
    ]
]


@dataclasses.dataclass
class FinalDecision:
    """Final row shape (pre-validation). Maps 1:1 onto output.csv columns --
    evidence_message_ids is already the semicolon-separated string / "none"
    here; validate.py only repairs it, it doesn't reshape it.
    """

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


def _requests_sensitive_action(text: str) -> bool:
    """True if text asks for an OTP/PIN/password/verification/payment or a
    link/button click -- the classic phishing payload that makes an
    injection_flag hit actually dangerous rather than just noisy.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in SENSITIVE_ACTION_PATTERNS)


def _has_real_business_relationship_evidence(evidence_ids: list[str], evidence_shortlist: list[dict]) -> bool:
    """True if any cited evidence_id maps to a past message this user
    actually opened or replied to from this sender/business -- real
    engagement, not just a row existing. Used to lift business_trust_floor
    when the agent grounded its recommendation in genuine history.
    """
    by_id = {e["message_id"]: e for e in evidence_shortlist}
    return any((entry := by_id.get(eid)) and (entry.get("opened") or entry.get("replied")) for eid in evidence_ids)


def _validate_message_type(guess: Optional[str]) -> str:
    return guess if guess in config.ALLOWED_MESSAGE_TYPES else "unknown"


def _agent_text(agent_output: AgentOutput) -> str:
    return f"{agent_output.reasoning} {agent_output.summary}"


def _has_low_urgency_language(agent_output: AgentOutput) -> bool:
    """True if the agent's own reasoning/summary admits this isn't urgent --
    trust the description over the recommendation when they disagree.
    """
    text = _agent_text(agent_output)
    return any(pattern.search(text) for pattern in LOW_URGENCY_PATTERNS)


def _has_high_urgency_language(agent_output: AgentOutput) -> bool:
    """True if the agent's reasoning/summary affirmatively justifies
    urgency -- the bar a group notify has to clear (see Fix 1b).

    Checks negation first: r"\burgent\b" matches inside "non-urgent" (the
    hyphen counts as a word boundary), which would otherwise false-positive
    "high urgency" out of a sentence that's actually saying the opposite.
    Any negated-urgency phrasing (already in LOW_URGENCY_PATTERNS) short-
    circuits this to False regardless of what the raw HIGH_URGENCY_PATTERNS
    regex would match.
    """
    if _has_low_urgency_language(agent_output):
        return False
    text = _agent_text(agent_output)
    return any(pattern.search(text) for pattern in HIGH_URGENCY_PATTERNS)


def _indicates_uncertainty(agent_output: AgentOutput) -> bool:
    """confidence_language is enum-validated to {high, medium, low} by
    tools.submit_understanding, so hedging words can't land there --
    check the free-text reasoning/summary instead, which is where a model
    that isn't sure actually says so.
    """
    text = f"{agent_output.confidence_language} {agent_output.reasoning} {agent_output.summary}".lower()
    return any(marker in text for marker in UNCERTAINTY_MARKERS)


def _calibrate_confidence(
    ctx: MessageContext,
    agent_output: AgentOutput,
    final_action: str,
    override_fired: bool,
    evidence_ids: list[str],
) -> float:
    """Architecture §6: code-computed confidence, base 0.85, nudged by
    agreement/evidence/uncertainty/history, then clipped to
    [config.CONFIDENCE_MIN, config.CONFIDENCE_MAX].
    """
    confidence = 0.85
    if override_fired:
        confidence -= 0.04
    else:
        confidence += 0.03
    if evidence_ids:
        confidence += 0.02
    if _indicates_uncertainty(agent_output):
        confidence -= 0.03
    if ctx.signals.first_time_sender:
        confidence -= 0.02
    if ctx.signals.repetition_prior > 0.5 and final_action == "mute":
        confidence += 0.03
    return max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, confidence))


def make_decision(agent_output: Optional[AgentOutput], ctx: MessageContext) -> FinalDecision:
    """Produce the final decision for one message.

    1. agent_output is None -> fixed fallback (digest/unknown/0.5/none).
    2. injection_flag + a sensitive-action request in the message text ->
       hard override to mute/scam, regardless of what the agent recommended.
    3. business_trust_floor with no cited evidence of real engagement ->
       hard override to mute/scam. Evidence of real engagement lifts the
       floor and the agent's recommendation is trusted as-is.
    4. Otherwise: start from agent_output.recommended_action and cascade the
       soft priors in order: mute_prior, opt_out_prior (promotion only --
       see the comment inline on why business_update was tried and reverted),
       high_forward_count, during_dnd, then three eval-driven consistency
       checks -- Fix 1a downgrades notify to digest when the agent's own
       reasoning admits low urgency, Fix 1b defaults group notify to digest
       absent a direct mention, an admin sender, or affirmative urgency
       language, and Fix 2a escalates digest to mute when
       repetition_prior > 0.5. first_time_sender never changes the action;
       cold_business_sender only adds a note.
    5. Finally (regardless of which branch above fired): Fix B's three
       message_type consistency corrections -- urgent+mute is contradictory
       (recategorized scam/spam), personal+business without an explicit
       personal justification (recategorized business_update), and a highly
       forwarded greeting (recategorized forward).
    """
    if agent_output is None:
        return FinalDecision(
            message_id=ctx.message_id,
            action="digest",
            message_type="unknown",
            reason=FALLBACK_REASON,
            confidence=FALLBACK_CONFIDENCE,
            evidence_message_ids="none",
        )

    signals = ctx.signals
    message_type = _validate_message_type(agent_output.message_type_guess)
    evidence_ids = list(agent_output.evidence_ids or [])
    evidence_message_ids = ";".join(evidence_ids) if evidence_ids else "none"

    action = agent_output.recommended_action
    overrides: list[str] = []
    notes: list[str] = []

    if signals.injection_flag and _requests_sensitive_action(ctx.message_text):
        action = "mute"
        message_type = "scam"
        overrides.append(
            f"prompt-injection pattern detected ({signals.injection_matched_pattern!r}) combined with a request "
            "for OTP/payment/verification/link click -- treated as a scam attempt regardless of the agent's recommendation"
        )
    elif signals.business_trust_floor and not _has_real_business_relationship_evidence(evidence_ids, ctx.evidence_shortlist):
        action = "mute"
        message_type = "scam"
        overrides.append(
            "business_trust_floor fired (unverified sender, domain mismatch, account under "
            f"{config.BUSINESS_TRUST_FLOOR_MAX_AGE} days old) and no cited evidence shows a real prior "
            "relationship (opened/replied) with this business"
        )
    else:
        if signals.mute_prior and not signals.direct_mention:
            if action == "notify":
                action = "digest"
                overrides.append("group is muted by the user and this message doesn't @mention them -- downgraded notify to digest")
        # mute_prior + direct_mention: explicit pass-through, a direct mention
        # overrides a muted group even for notify -- no code change needed.

        if signals.opt_out_prior and message_type == "promotion":
            # NOT extended to business_update: real businesses send both
            # promotional AND transactional messages under one account, and
            # "opted out of promotions" means the user doesn't want sale
            # posters, not that they want to miss a delivery notification.
            # A prior attempt to broaden this to business_update caused 3
            # regressions (order/health/feedback updates wrongly muted) and
            # 0 wins -- reverted.
            if action != "mute":
                overrides.append("user opted out of promotions from this business and the message is promotional -- forced mute")
            action = "mute"
            message_type = "promotion"

        if signals.high_forward_count and message_type in ("greeting", "forward"):
            if action != "mute":
                overrides.append(f"forwarded_count={ctx.forwarded_count} on a {message_type} message -- biased toward mute")
            action = "mute"

        if signals.during_dnd and agent_output.urgency != "high":
            if action == "notify":
                action = "digest"
                overrides.append("message arrived during the user's do-not-disturb window and urgency isn't high -- downgraded notify to digest")

        # Fix 1a (30-sample eval finding): the agent's own reasoning admits
        # low urgency but it recommended notify anyway -- trust the
        # description over the recommendation. Exempts direct_mention (same
        # gap Fix 1b already accounted for): a message that directly
        # @mentions this user is notify-worthy even phrased casually --
        # a prior version without this exemption wrongly downgraded a
        # direct personal ask ("nothing dramatic... non-urgent") that
        # ground truth says should notify (sample_msg_006).
        if action == "notify" and not signals.direct_mention and _has_low_urgency_language(agent_output):
            action = "digest"
            overrides.append("agent recommended notify but its own reasoning/summary contains low-urgency language -- downgraded to digest")

        # Fix 1b: a group message with no direct mention and no mute_prior
        # involvement needs an affirmative urgency justification to earn
        # notify -- absent one, default to digest rather than assuming
        # legitimate/relevant content is automatically interrupt-worthy.
        # Exempted when the sender is a group admin: admin-sourced actionable
        # content (consent forms, deadlines, notices) is inherently higher
        # priority than a regular member's message even without urgency
        # wording -- a prior version without this exemption wrongly
        # downgraded a school admin's consent-form notice (sample_msg_046).
        if (
            action == "notify"
            and ctx.conversation_type == "group"
            and not signals.mute_prior
            and not signals.direct_mention
            and not signals.sender_is_group_admin
            and not _has_high_urgency_language(agent_output)
        ):
            action = "digest"
            overrides.append("group message with no direct mention, no admin sender, and no explicit urgency language in the agent's reasoning -- defaulted to digest instead of notify")

        # Fix 2a: strong historical signal (>50% of this sender's past
        # messages were dismissed/muted/reported) but the agent only
        # recommended digest -- the user has already told us they don't
        # want this, so escalate.
        if signals.repetition_prior > 0.5 and action == "digest":
            action = "mute"
            overrides.append(f"repetition_prior={signals.repetition_prior:.2f} (>0.5) shows a consistent pattern of the user dismissing/muting/reporting this sender -- escalated digest to mute")

        # first_time_sender with no risk signals: agent's call stands, no override.
        if signals.cold_business_sender:
            notes.append("no user_business_history relationship on file for this business")

    # Fix B (30-sample eval finding): a few adjacent-category message_type
    # corrections, applied regardless of which branch above produced the
    # current action/message_type -- these catch the agent's guess
    # conflicting with the final action or with other known signals.
    if action == "mute" and message_type == "urgent":
        message_type = "scam" if signals.injection_flag else "spam"
        overrides.append("message_type_guess was 'urgent' but the final action is mute -- urgent+mute is contradictory, recategorized")

    if ctx.conversation_type == "business" and message_type == "personal" and "personal" not in agent_output.reasoning.lower():
        message_type = "business_update"
        overrides.append("message_type_guess was 'personal' for a business conversation with no explicit personal justification in the agent's reasoning -- recategorized as business_update")

    if ctx.forwarded_count >= 5 and message_type == "greeting":
        message_type = "forward"
        overrides.append(f"forwarded_count={ctx.forwarded_count} (>=5) on a greeting -- recategorized as forward")

    override_fired = action != agent_output.recommended_action
    confidence = _calibrate_confidence(ctx, agent_output, action, override_fired, evidence_ids)

    reason_parts = []
    if overrides:
        reason_parts.append("Override: " + " | ".join(overrides) + ".")
    if notes:
        reason_parts.append("Note: " + " | ".join(notes) + ".")
    reason_parts.append(f"Agent reasoning: {agent_output.reasoning}")

    return FinalDecision(
        message_id=ctx.message_id,
        action=action,
        message_type=message_type,
        reason=" ".join(reason_parts),
        confidence=confidence,
        evidence_message_ids=evidence_message_ids,
    )


def _integration_test() -> None:
    """Full DESCRIBE -> DECIDE -> VALIDATE chain on 3 sample messages (one
    per conversation_type, including the prompt-injection case), printing
    the final output row for each next to its ground-truth label.
    """
    import csv

    from agent import run_agent
    from context import build_context, load_dataset
    from validate import validate_output

    dataset = load_dataset()
    with config.SAMPLE_MESSAGES_CSV.open(newline="", encoding="utf-8-sig") as f:
        sample_rows = {row["message_id"]: row for row in csv.DictReader(f)}

    for message_id in ["sample_msg_004", "sample_msg_001", "sample_msg_053"]:
        row = sample_rows[message_id]
        # get_evidence_candidates looks up dataset.messages_by_id (built from
        # messages.csv); sample_messages.csv ids aren't in there. Any real
        # routed message IS in messages.csv already, so this only matters
        # for this standalone sample-based test.
        dataset.messages_by_id[message_id] = row
        ctx = build_context(row, dataset)

        agent_output = run_agent(ctx, dataset)
        decision = make_decision(agent_output, ctx)
        validated = validate_output(decision, ctx.evidence_shortlist)

        print(f"\n{'=' * 80}")
        print(f"{message_id} -- ground truth: action={row.get('action')}, message_type={row.get('message_type')}")
        print("=" * 80)
        print(f"  action:               {validated.action}")
        print(f"  message_type:         {validated.message_type}")
        print(f"  confidence:           {validated.confidence}")
        print(f"  evidence_message_ids: {validated.evidence_message_ids}")
        print(f"  reason:               {validated.reason}")


if __name__ == "__main__":
    _integration_test()
