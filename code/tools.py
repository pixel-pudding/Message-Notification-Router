"""Stage 3 (locked architecture §4): TOOLS.

Each tool is a plain Python function -- (id[, extra ids], dataset) -> dict
(or list[dict] for get_evidence_candidates) -- with no side effects and no
model calls. They exist so the agent (stage 4, still unbuilt) can pull in
exactly the context it decides it needs, instead of every field being
force-fed into every prompt. TOOL_SCHEMAS is the OpenAI-compatible
function-calling declaration of the same six lookup tools plus the special
submit_understanding tool that ends the agent's turn; TOOL_FUNCTIONS maps
each schema name to its Python callable for the future dispatch loop.

Every lookup tool takes a plain `dataset: context.Dataset` (from
context.load_dataset()) rather than holding its own state -- there is
nothing here to construct or configure, just call the functions directly.

Run standalone for a smoke test against real dataset ids:

    python code/tools.py
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import config
import media
from context import Dataset, build_context, compute_business_trust_floor

# --- shared parsing helpers (CSV cells are always strings) ---


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _to_bool(value: Optional[str]) -> bool:
    return value == "1"


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    return value if value else None


# --- 1. get_user_profile ---


def get_user_profile(user_id: str, dataset: Dataset) -> dict[str, Any]:
    """User-level notification behavior: DND window, 30-day engagement
    counters, and recent daily notification load (avg notifications_sent
    over that user's last 7 rows in daily_notification_summary.csv).
    """
    user = dataset.users_by_id.get(user_id)
    if user is None:
        return {"status": "not_found"}

    daily_rows = dataset.daily_notification_by_user.get(user_id, [])
    recent = sorted(daily_rows, key=lambda r: r["date"], reverse=True)[:7]
    recent_daily_notification_load = (
        round(sum(int(r["notifications_sent"]) for r in recent) / len(recent), 2) if recent else None
    )

    return {
        "do_not_disturb_window": user.get("do_not_disturb_window"),
        "messages_opened_30d": _to_int(user.get("messages_opened_30d")),
        "messages_replied_30d": _to_int(user.get("messages_replied_30d")),
        "notifications_dismissed_30d": _to_int(user.get("notifications_dismissed_30d")),
        "messages_reported_30d": _to_int(user.get("messages_reported_30d")),
        "recent_daily_notification_load": recent_daily_notification_load,
    }


# --- 2. get_group_context ---


def get_group_context(group_id: str, user_id: str, dataset: Dataset) -> dict[str, Any]:
    """Group metadata plus this specific user's role and activity within it
    (including whether they've muted the group). Returns status="not_found"
    if the group doesn't exist or this user isn't a member of it.
    """
    group = dataset.groups_by_id.get(group_id)
    member = dataset.group_members_by_key.get((group_id, user_id))
    if group is None or member is None:
        return {"status": "not_found"}

    return {
        "group_name": group.get("group_name"),
        "group_type": group.get("group_type"),
        "member_count": _to_int(group.get("member_count")),
        "admin_count": _to_int(group.get("admin_count")),
        "messages_30d": _to_int(group.get("messages_30d")),
        "user_role": member.get("role"),
        "user_messages_sent_30d": _to_int(member.get("messages_sent_30d")),
        "user_messages_read_30d": _to_int(member.get("messages_read_30d")),
        "user_replies_sent_30d": _to_int(member.get("replies_sent_30d")),
        "user_notifications_dismissed_30d": _to_int(member.get("notifications_dismissed_30d")),
        "group_muted_by_user": _to_bool(member.get("group_muted_by_user")),
    }


# --- 3. get_business_profile ---


def get_business_profile(business_id: str, dataset: Dataset) -> dict[str, Any]:
    """Business sender identity/verification/reputation, plus
    business_trust_floor -- the same unverified + domain-mismatch +
    young-account check context.py's pre-gate uses, computed identically
    here so the agent's read matches the deterministic gate exactly.
    """
    business = dataset.business_by_id.get(business_id)
    if business is None:
        return {"status": "not_found"}

    return {
        "display_name": business.get("display_name"),
        "brand_name": business.get("brand_name"),
        "category": business.get("category"),
        "verified": _to_bool(business.get("verified")),
        "official_domain": _none_if_blank(business.get("official_domain")),
        "domain_used_by_sender": _none_if_blank(business.get("domain_used_by_sender")),
        "account_age_days": _to_int(business.get("account_age_days")),
        "messages_sent_30d": _to_int(business.get("messages_sent_30d")),
        "user_reports_30d": _to_int(business.get("user_reports_30d")),
        "domain_used_by_sender_age_days": _to_int(business.get("domain_used_by_sender_age_days")),
        "business_trust_floor": compute_business_trust_floor("business", business),
    }


# --- 4. get_user_business_history ---


def get_user_business_history(user_id: str, business_id: str, dataset: Dataset) -> dict[str, Any]:
    """This specific user's relationship with this specific business:
    how they know the account, promo opt-in/out, and 30/180-day activity.
    Returns status="no_relationship" if there's no row for this pair -- a
    real, common case (about a third of business messages in this dataset
    have no prior relationship on file), not an error.
    """
    row = dataset.user_business_by_key.get((user_id, business_id))
    if row is None:
        return {"status": "no_relationship"}

    return {
        "why_user_knows_account": row.get("why_user_knows_account"),
        "last_activity_at": _none_if_blank(row.get("last_activity_at")),
        "allows_promotions": _to_bool(row.get("allows_promotions")),
        "promotions_opted_out_at": _none_if_blank(row.get("promotions_opted_out_at")),
        "activity_count_180d": _to_int(row.get("activity_count_180d")),
        "messages_opened_30d": _to_int(row.get("messages_opened_30d")),
        "messages_dismissed_30d": _to_int(row.get("messages_dismissed_30d")),
        "messages_replied_30d": _to_int(row.get("messages_replied_30d")),
        "last_reply_at": _none_if_blank(row.get("last_reply_at")),
    }


# --- 5. get_evidence_candidates ---


def get_evidence_candidates(message_id: str, dataset: Dataset) -> list[dict[str, Any]]:
    """The pre-retrieved evidence shortlist (context.py stage 3) for this
    incoming message_id: up to 8 past messages from the same sender,
    business, or group, most recent first, each with its message_events
    outcome. Returns [] if message_id is unknown or no candidates exist --
    that's a legitimate result (write evidence_message_ids="none"), not a
    signal to keep searching.
    """
    message_row = dataset.messages_by_id.get(message_id)
    if message_row is None:
        return []

    ctx = build_context(message_row, dataset)
    return [
        {
            "message_id": e["message_id"],
            "text_snippet": e["text_snippet"],
            "sender": e["sender"],
            "timestamp": e["created_at"],
            "event_outcomes": {
                "opened": e["opened"],
                "replied": e["replied"],
                "dismissed": e["dismissed"],
                "muted": e["muted_after"],
                "reported": e["reported"],
            },
        }
        for e in ctx.evidence_shortlist
    ]


# --- 6. get_media_text ---


def get_media_text(media_id: str, dataset: Dataset) -> dict[str, Any]:
    """OCR text + classification for an image, or transcript for a voice
    note, from the (already-preprocessed, disk-cached) media layer --
    code/media.py. Returns status="no_media" if media_id is unknown or
    hasn't been preprocessed yet; run `python code/media.py` first to
    populate the cache. `dataset` is accepted for signature consistency
    with the other tools but isn't needed here -- the media cache is
    self-contained.
    """
    del dataset  # unused: media cache is independent of the CSV dataset
    record = media.get_media_record(media_id)
    if record is None:
        return {"status": "no_media"}
    if isinstance(record, media.ImageAnalysis):
        return {
            "media_type": "image",
            "content": record.visible_text,
            "image_type": record.image_type,
            "flags": record.flags,
        }
    return {
        "media_type": "voice",
        "content": record.transcript,
    }


# --- 7. submit_understanding (ends the agent's turn) ---

VALID_URGENCY_LEVELS = {"high", "medium", "low", "none"}
VALID_CONFIDENCE_LANGUAGE = {"high", "medium", "low"}


def submit_understanding(
    summary: str,
    intent: str,
    message_type_guess: str,
    urgency: str,
    risk_flags: list[str],
    recommended_action: str,
    evidence_ids: list[str],
    confidence_language: str,
    reasoning: str,
) -> dict[str, Any]:
    """Package (and validate) the agent's final structured understanding of
    one message. This is the only tool that doesn't look up data -- calling
    it ends the agent's turn. Every field here is a non-binding
    *observation*: DECIDE (decide.py) combines it with the pre-gate signals
    and engagement stats to compute the real action/message_type/confidence,
    so the agent should describe what it sees, not guess the final verdict.
    """
    errors: list[str] = []
    if message_type_guess not in config.ALLOWED_MESSAGE_TYPES:
        errors.append(f"message_type_guess must be one of {sorted(config.ALLOWED_MESSAGE_TYPES)}, got {message_type_guess!r}")
    if urgency not in VALID_URGENCY_LEVELS:
        errors.append(f"urgency must be one of {sorted(VALID_URGENCY_LEVELS)}, got {urgency!r}")
    if recommended_action not in config.ALLOWED_ACTIONS:
        errors.append(f"recommended_action must be one of {sorted(config.ALLOWED_ACTIONS)}, got {recommended_action!r}")
    if confidence_language not in VALID_CONFIDENCE_LANGUAGE:
        errors.append(f"confidence_language must be one of {sorted(VALID_CONFIDENCE_LANGUAGE)}, got {confidence_language!r}")
    if not isinstance(risk_flags, list) or not all(isinstance(x, str) for x in risk_flags):
        errors.append("risk_flags must be a list of strings")
    if not isinstance(evidence_ids, list) or not all(isinstance(x, str) for x in evidence_ids):
        errors.append("evidence_ids must be a list of message_id strings")

    result = {
        "summary": summary,
        "intent": intent,
        "message_type_guess": message_type_guess,
        "urgency": urgency,
        "risk_flags": risk_flags,
        "recommended_action": recommended_action,
        "evidence_ids": evidence_ids,
        "confidence_language": confidence_language,
        "reasoning": reasoning,
    }
    if errors:
        result["validation_errors"] = errors
    return result


# --- OpenAI-compatible function-calling schemas ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": (
                "Look up the RECEIVING user's own notification behavior: their do-not-disturb window, "
                "how often they open/reply/dismiss/report messages in the last 30 days, and their recent "
                "daily notification load (are they currently getting flooded?). Call this once per message "
                "to calibrate how interruptible this specific user is right now -- a user who dismisses "
                "most notifications and is already overloaded should lean toward digest/mute even for "
                "borderline-useful content; a highly responsive, lightly-loaded user can tolerate more notifies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The receiving user's id, e.g. 'u_002'."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_group_context",
            "description": (
                "Look up a group chat's metadata (name, type -- family/society/school_group/coworker/"
                "marketplace/etc, size, activity volume) AND this specific user's relationship to it "
                "(their role, how much they read/reply/dismiss, and whether they've muted the group). "
                "Call this for any conversation_type='group' message to judge whether the group itself "
                "is high-signal (e.g. small family group) or high-noise (e.g. large marketplace group), "
                "and whether a muted group should still be treated as opted-out of interruption."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "description": "The group's id, e.g. 'group_005'."},
                    "user_id": {"type": "string", "description": "The receiving user's id, to look up their membership row."},
                },
                "required": ["group_id", "user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_business_profile",
            "description": (
                "Look up a business sender's identity and reputation: display/brand name, category, "
                "verification status, official vs. actually-used sending domain, account age, and how many "
                "users have reported it recently. Also returns business_trust_floor, a precomputed bool that "
                "is True only when the business is unverified AND its sending domain doesn't match (or lacks) "
                "an official domain AND the account is under 60 days old -- treat business_trust_floor=True as "
                "a strong signal toward mute/scam regardless of message content. Call this for any "
                "conversation_type='business' message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "business_id": {"type": "string", "description": "The business account's id, e.g. 'business_001'."},
                },
                "required": ["business_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_business_history",
            "description": (
                "Look up whether THIS user has any prior relationship with THIS business: how they know the "
                "account (e.g. recent order, active subscription, upcoming appointment), whether they still "
                "allow promotional messages or have opted out, and their 30/180-day activity with this "
                "business. Returns {\"status\": \"no_relationship\"} if there's no row for this pair -- that is "
                "common and legitimate (about a third of business messages here have no history), and on its "
                "own is a reason to be more cautious, not an error to retry. Call this alongside "
                "get_business_profile for any conversation_type='business' message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The receiving user's id."},
                    "business_id": {"type": "string", "description": "The business account's id."},
                },
                "required": ["user_id", "business_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evidence_candidates",
            "description": (
                "Fetch up to 8 past messages relevant to the CURRENT message being routed -- same sender, "
                "same business, or same group, sent to this same user, most recent first -- each with a text "
                "snippet and how the user reacted to it (opened/replied/dismissed/muted/reported). Use this "
                "to check whether the user has a track record of ignoring or reporting this kind of message "
                "(supports mute/digest) or reliably engaging with it (supports notify), and to pick concrete "
                "message_ids for evidence_ids. Returns [] if there is no relevant history -- report that as "
                "no evidence found, don't invent message_ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "The id of the message currently being routed, e.g. 'msg_023'."},
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_media_text",
            "description": (
                "Read the OCR text + image classification/flags for an image message, or the transcript for "
                "a voice-note message, from the pre-processed media cache. ALWAYS call this when the current "
                "message has a media_id -- message_text alone is empty or incomplete for media messages, and "
                "you cannot judge urgency/risk/content without it. Returns {\"status\": \"no_media\"} if the "
                "id is unknown or hasn't been preprocessed; treat that as missing information, not as 'no text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "media_id": {"type": "string", "description": "The image_id or voice_note_id referenced by the current message, e.g. 'img_008' or 'vn_001'."},
                },
                "required": ["media_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_understanding",
            "description": (
                "End your turn by submitting your structured understanding of the current message. This is "
                "the ONLY way to finish -- do not just describe your conclusion in plain text. Every field is "
                "your OBSERVATION, not the final verdict: a separate deterministic step combines this with "
                "pre-gate signals and engagement history to compute the actual action/message_type/confidence "
                "that gets written to output.csv, so describe what you see (urgency, risk, intent) rather than "
                "trying to predict the final label. Call the lookup tools you need FIRST -- get_user_profile "
                "and, depending on conversation_type, get_group_context / get_business_profile+"
                "get_user_business_history, get_evidence_candidates, and get_media_text if there's a media_id "
                "-- then call this exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One or two plain-language sentences describing what this message actually is and why it was sent.",
                    },
                    "intent": {
                        "type": "string",
                        "description": "The sender's apparent goal in one short phrase, e.g. 'ask for urgent help', 'sell a used item', 'promote a discount', 'phish for OTP'.",
                    },
                    "message_type_guess": {
                        "type": "string",
                        "enum": sorted(config.ALLOWED_MESSAGE_TYPES),
                        "description": "Best-fit category for this message. Non-binding -- DECIDE may override it using pre-gate signals.",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": sorted(VALID_URGENCY_LEVELS),
                        "description": "How time-sensitive this message is for the receiving user specifically -- 'high' means it needs action within minutes/hours, 'none' means there's no time pressure at all.",
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short tags for any safety/trust concerns you noticed, e.g. ['otp_request', 'suspicious_link', 'urgency_pressure', 'unverified_sender']. Empty list if none.",
                    },
                    "recommended_action": {
                        "type": "string",
                        "enum": sorted(config.ALLOWED_ACTIONS),
                        "description": "Your recommended routing action given everything you looked up. Non-binding -- DECIDE makes the final call and may not agree with you, especially when a pre-gate signal fired.",
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "message_id values FROM get_evidence_candidates results that actually support your reasoning. Do not invent ids; use [] if get_evidence_candidates returned nothing useful.",
                    },
                    "confidence_language": {
                        "type": "string",
                        "enum": sorted(VALID_CONFIDENCE_LANGUAGE),
                        "description": "Roughly how confident you are in this understanding, in words (not a number) -- DECIDE converts this into a calibrated numeric confidence.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A short chain of reasoning connecting what you looked up to your conclusion -- this feeds into the final human-readable 'reason' column.",
                    },
                },
                "required": [
                    "summary",
                    "intent",
                    "message_type_guess",
                    "urgency",
                    "risk_flags",
                    "recommended_action",
                    "evidence_ids",
                    "confidence_language",
                    "reasoning",
                ],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_user_profile": get_user_profile,
    "get_group_context": get_group_context,
    "get_business_profile": get_business_profile,
    "get_user_business_history": get_user_business_history,
    "get_evidence_candidates": get_evidence_candidates,
    "get_media_text": get_media_text,
    "submit_understanding": submit_understanding,
}


def _smoke_test() -> None:
    from context import load_dataset

    dataset = load_dataset()

    print("--- get_user_profile('u_002') ---")
    print(get_user_profile("u_002", dataset))

    print("\n--- get_group_context('group_002', 'u_011') ---")
    print(get_group_context("group_002", "u_011", dataset))

    print("\n--- get_group_context('group_002', 'u_999') [not found] ---")
    print(get_group_context("group_002", "u_999", dataset))

    print("\n--- get_business_profile('business_002') ---")
    print(get_business_profile("business_002", dataset))

    print("\n--- get_user_business_history('u_002', 'business_002') ---")
    print(get_user_business_history("u_002", "business_002", dataset))

    print("\n--- get_user_business_history('u_001', 'business_999') [no relationship] ---")
    print(get_user_business_history("u_001", "business_999", dataset))

    print("\n--- get_evidence_candidates('msg_023') ---")
    for item in get_evidence_candidates("msg_023", dataset):
        print(item)

    print("\n--- get_media_text('img_008', dataset) ---")
    print(get_media_text("img_008", dataset))

    print("\n--- get_media_text('vn_001', dataset) ---")
    print(get_media_text("vn_001", dataset))

    print("\n--- get_media_text('img_999', dataset) [no cache entry] ---")
    print(get_media_text("img_999", dataset))

    print("\n--- submit_understanding(...) [valid] ---")
    print(
        submit_understanding(
            summary="A bank sends a routine card-payment update the user has seen and opened before.",
            intent="inform about account update",
            message_type_guess="business_update",
            urgency="low",
            risk_flags=[],
            recommended_action="digest",
            evidence_ids=["message_0243"],
            confidence_language="high",
            reasoning="Near-identical past message from the same verified business was opened and replied to; no urgency or risk markers.",
        )
    )

    print("\n--- submit_understanding(...) [invalid enum values] ---")
    print(
        submit_understanding(
            summary="test",
            intent="test",
            message_type_guess="not_a_real_type",
            urgency="extremely-high",
            risk_flags="not_a_list",
            recommended_action="ignore",
            evidence_ids=["message_0001"],
            confidence_language="very sure",
            reasoning="test",
        )
    )

    print(f"\n{len(TOOL_SCHEMAS)} tool schemas defined: {[s['function']['name'] for s in TOOL_SCHEMAS]}")


if __name__ == "__main__":
    _smoke_test()
