"""Stage 1: LOAD & JOIN, plus hard signal computation (architecture doc
pipeline stage 2) and evidence shortlist retrieval (stage 3).

Reads every dataset/*.csv file once, builds lookup indices, and produces a
fully joined MessageContext for each incoming message in messages.csv. This
is the only stage that touches raw CSV files directly -- every later stage
(pregate, agent, decide, validate) consumes MessageContext objects, not CSVs.
It also reads from the media cache (code/media.py) to fold OCR/ASR text into
the injection-detection signal, but never calls a vision/ASR/routing model
itself.

Run this file directly to print the joined context for a few sample
messages, e.g.:

    python code/context.py --sample msg_023 msg_005 msg_086

Or run the hard-signal test harness over dataset/sample_messages.csv:

    python code/context.py --signal-test
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from datetime import datetime, time
from pathlib import Path
from typing import Iterator, Optional

import config
import media

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

HISTORY_LIMIT_DEFAULT = 5
EVIDENCE_SHORTLIST_LIMIT = 8


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of plain dicts (all values kept as strings)."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    """CSV empty cells come back as '' -- normalize to None so joins/checks are unambiguous."""
    return value if value else None


@dataclasses.dataclass
class Dataset:
    """All raw dataset tables plus the indices build_context() joins against."""

    dataset_dir: Path
    messages: list[dict]
    messages_by_id: dict[str, dict]
    users_by_id: dict[str, dict]
    groups_by_id: dict[str, dict]
    group_members_by_key: dict[tuple[str, str], dict]  # (group_id, user_id) -> row
    business_by_id: dict[str, dict]
    user_business_by_key: dict[tuple[str, str], dict]  # (user_id, business_id) -> row
    message_history: list[dict]
    message_events_by_message_id: dict[str, dict]
    images_by_id: dict[str, dict]
    voice_notes_by_id: dict[str, dict]
    daily_notification_by_user: dict[str, list[dict]]  # user_id -> rows, file order


def load_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Dataset:
    """Read every dataset/*.csv file needed for joining and build the lookup indices.

    Called once per run (see main.py). Everything downstream operates on the
    returned Dataset plus per-message MessageContext objects -- no repeated
    disk I/O per message.
    """
    messages = _read_csv(dataset_dir / "messages.csv")
    users = _read_csv(dataset_dir / "users.csv")
    groups = _read_csv(dataset_dir / "groups.csv")
    group_members = _read_csv(dataset_dir / "group_members.csv")
    business_accounts = _read_csv(dataset_dir / "business_accounts.csv")
    user_business_history = _read_csv(dataset_dir / "user_business_history.csv")
    message_history = _read_csv(dataset_dir / "message_history.csv")
    message_events = _read_csv(dataset_dir / "message_events.csv")
    images = _read_csv(dataset_dir / "images.csv")
    voice_notes = _read_csv(dataset_dir / "voice_notes.csv")
    daily_notification_summary = _read_csv(dataset_dir / "daily_notification_summary.csv")

    daily_notification_by_user: dict[str, list[dict]] = {}
    for row in daily_notification_summary:
        daily_notification_by_user.setdefault(row["user_id"], []).append(row)

    return Dataset(
        dataset_dir=dataset_dir,
        messages=messages,
        messages_by_id={row["message_id"]: row for row in messages},
        users_by_id={row["user_id"]: row for row in users},
        groups_by_id={row["group_id"]: row for row in groups},
        group_members_by_key={
            (row["group_id"], row["user_id"]): row for row in group_members
        },
        business_by_id={row["business_id"]: row for row in business_accounts},
        user_business_by_key={
            (row["user_id"], row["business_id"]): row for row in user_business_history
        },
        message_history=message_history,
        message_events_by_message_id={row["message_id"]: row for row in message_events},
        images_by_id={row["image_id"]: row for row in images},
        voice_notes_by_id={row["voice_note_id"]: row for row in voice_notes},
        daily_notification_by_user=daily_notification_by_user,
    )


@dataclasses.dataclass
class HistoricalMessage:
    """One past message from the same sender/business to this user, plus how the user reacted to it (if recorded in message_events.csv)."""

    message_id: str
    created_at: str
    message_text: str
    media_type: Optional[str]
    forwarded_count: int
    sender_user_id: Optional[str]
    business_id: Optional[str]
    opened: Optional[bool]
    replied: Optional[bool]
    dismissed: Optional[bool]
    muted_after: Optional[bool]
    reported: Optional[bool]
    reaction_time_minutes: Optional[int]


@dataclasses.dataclass
class HardSignals:
    """Deterministic, model-free signals computed once per message (locked
    architecture doc, pipeline stage 2). Reused as-is by the pre-gate
    (stage 2 checks) and by DECIDE (stage 5 calibration) -- neither stage
    recomputes these from raw CSVs.
    """

    injection_flag: bool
    injection_matched_pattern: Optional[str]
    business_trust_floor: bool
    repetition_prior: float
    mute_prior: bool
    opt_out_prior: bool
    direct_mention: bool
    first_time_sender: bool
    high_forward_count: bool
    during_dnd: bool
    cold_business_sender: bool
    sender_is_group_admin: bool  # role=="admin" for sender_user_id in THIS group (not the receiver's own role)


@dataclasses.dataclass
class MessageContext:
    """Everything known about one incoming message, joined and ready for the
    pre-gate, model, and decision stages. Field groups mirror the LOAD & JOIN
    pipeline stage's join list, plus the hard-signal/evidence-shortlist
    fields computed on top of that join.
    """

    # --- raw message fields (messages.csv) ---
    message_id: str
    user_id: str
    conversation_type: str  # personal | group | business
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: str
    message_text: str
    media_type: Optional[str]  # "" | image | voice
    media_id: Optional[str]
    forwarded_count: int

    # --- joined context ---
    user: Optional[dict]  # users.csv row for user_id
    group: Optional[dict]  # groups.csv row for group_id, if conversation_type == group
    group_member: Optional[dict]  # group_members.csv row for (group_id, user_id)
    business: Optional[dict]  # business_accounts.csv row for business_id, if conversation_type == business
    user_business: Optional[dict]  # user_business_history.csv row for (user_id, business_id)
    image: Optional[dict]  # images.csv row for media_id, if media_type == image
    voice_note: Optional[dict]  # voice_notes.csv row for media_id, if media_type == voice
    relevant_history: list[HistoricalMessage]  # past messages from this sender/business to this user (narrow match, capped at history_limit)

    # --- stage 2/3: hard signals + evidence shortlist ---
    signals: HardSignals
    evidence_shortlist: list[dict]  # broad match (sender/business/group), capped at EVIDENCE_SHORTLIST_LIMIT, dict-shaped for the model prompt


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value == "1"


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def lookup_relevant_history(
    dataset: Dataset,
    user_id: str,
    sender_user_id: Optional[str],
    business_id: Optional[str],
    group_id: Optional[str] = None,
    limit: Optional[int] = HISTORY_LIMIT_DEFAULT,
) -> list[HistoricalMessage]:
    """Find past messages to `user_id` from the same `business_id` (if set),
    the same `sender_user_id` (if set), or -- when group_id is passed -- the
    same `group_id`, most recent first, joined with their message_events.csv
    outcome. group_id is opt-in: the hard signals (repetition_prior,
    first_time_sender) intentionally use the narrower sender/business-only
    match, while the evidence shortlist (build_evidence_shortlist) opts into
    the broader group match. limit=None returns every match, uncapped.
    Shared by build_context() and tools.lookup_history() so the pre-computed
    context and the agent's on-demand tool call agree.
    """
    if not business_id and not sender_user_id and not group_id:
        return []

    matches = [
        row
        for row in dataset.message_history
        if row["user_id"] == user_id
        and (
            (business_id and row.get("business_id") == business_id)
            or (sender_user_id and row.get("sender_user_id") == sender_user_id)
            or (group_id and row.get("group_id") == group_id)
        )
    ]
    matches.sort(key=lambda row: row["created_at"], reverse=True)
    if limit is not None:
        matches = matches[:limit]

    history: list[HistoricalMessage] = []
    for row in matches:
        event = dataset.message_events_by_message_id.get(row["message_id"])
        history.append(
            HistoricalMessage(
                message_id=row["message_id"],
                created_at=row["created_at"],
                message_text=row.get("message_text", ""),
                media_type=_none_if_blank(row.get("media_type")),
                forwarded_count=_parse_int(row.get("forwarded_count")) or 0,
                sender_user_id=_none_if_blank(row.get("sender_user_id")),
                business_id=_none_if_blank(row.get("business_id")),
                opened=_parse_bool(event.get("message_opened")) if event else None,
                replied=_parse_bool(event.get("message_replied")) if event else None,
                dismissed=_parse_bool(event.get("notification_dismissed")) if event else None,
                muted_after=_parse_bool(event.get("muted_after_message")) if event else None,
                reported=_parse_bool(event.get("message_reported")) if event else None,
                reaction_time_minutes=_parse_int(event.get("reaction_time_minutes")) if event else None,
            )
        )
    return history


def safe_media_text(media_id: Optional[str]) -> str:
    """Look up OCR/ASR text for media_id from the (disk-persisted) media
    cache. Returns "" if media_id is None or hasn't been preprocessed yet --
    context building must never fail just because media.py hasn't run.
    """
    if not media_id:
        return ""
    try:
        return media.get_media_text(media_id)
    except KeyError:
        return ""


def detect_injection(*texts: str) -> tuple[bool, Optional[str]]:
    """Run config.INJECTION_PATTERNS against the given text fields (message
    text, media OCR/ASR text). Returns (True, matched_pattern) on the first
    hit, else (False, None).
    """
    combined = "\n".join(t for t in texts if t)
    for pattern in config.INJECTION_PATTERNS:
        if pattern.search(combined):
            return True, pattern.pattern
    return False, None


def compute_business_trust_floor(conversation_type: str, business: Optional[dict]) -> bool:
    """True only for conversation_type == "business" when the business is
    unverified, its sending domain doesn't match (or there is no) official
    domain, AND the account is younger than config.BUSINESS_TRUST_FLOOR_MAX_AGE
    days. All three must hold -- a young but verified/domain-matched business
    does not trip this.
    """
    if conversation_type != "business" or business is None:
        return False
    if business.get("verified") != "0":
        return False
    official_domain = _none_if_blank(business.get("official_domain"))
    domain_used = _none_if_blank(business.get("domain_used_by_sender"))
    domain_mismatch = official_domain is None or domain_used != official_domain
    if not domain_mismatch:
        return False
    age = _parse_int(business.get("account_age_days"))
    return age is not None and age < config.BUSINESS_TRUST_FLOOR_MAX_AGE


def compute_repetition_prior(history: list[HistoricalMessage]) -> float:
    """Ratio of (dismissed + muted_after + reported) counts to the number of
    historical messages with a recorded event, across ALL matching past
    messages from this sender/business to this user (not just the capped
    display slice) -- a higher ratio means the user has a track record of
    ignoring or reporting this sender. 0.0 if no prior events exist. Can
    exceed 1.0 when a single event trips more than one of the three flags;
    that's fine, this is a bias signal for DECIDE, not a probability.
    """
    with_events = [h for h in history if h.opened is not None]
    if not with_events:
        return 0.0
    negative = sum(
        int(bool(h.dismissed)) + int(bool(h.muted_after)) + int(bool(h.reported))
        for h in with_events
    )
    return negative / len(with_events)


def compute_direct_mention(message_text: str, user_id: str) -> bool:
    """True if message_text @-mentions this exact user_id (optional space
    after the @, word boundary after the id so "@u_010" doesn't also match
    a longer id like "@u_0105").
    """
    if not message_text:
        return False
    pattern = re.compile(rf"@\s*{re.escape(user_id)}\b")
    return bool(pattern.search(message_text))


def _parse_dnd_window(window: Optional[str]) -> Optional[tuple[time, time]]:
    if not window or "-" not in window:
        return None
    start_str, end_str = window.split("-", 1)
    try:
        start = datetime.strptime(start_str.strip(), "%H:%M").time()
        end = datetime.strptime(end_str.strip(), "%H:%M").time()
    except ValueError:
        return None
    return start, end


def compute_during_dnd(created_at: str, do_not_disturb_window: Optional[str]) -> bool:
    """True if created_at's time-of-day falls inside do_not_disturb_window
    (e.g. "22:00-07:00"), handling the overnight wrap-around where the
    window crosses midnight (start > end).
    """
    parsed_window = _parse_dnd_window(do_not_disturb_window)
    if parsed_window is None:
        return False
    start, end = parsed_window
    try:
        msg_time = datetime.strptime(created_at, "%Y-%m-%d %H:%M").time()
    except ValueError:
        return False
    if start <= end:
        return start <= msg_time <= end
    return msg_time >= start or msg_time <= end


def compute_hard_signals(
    *,
    message_text: str,
    media_text: str,
    conversation_type: str,
    user_id: str,
    created_at: str,
    forwarded_count: int,
    user: Optional[dict],
    business: Optional[dict],
    user_business: Optional[dict],
    group_member: Optional[dict],
    sender_group_member: Optional[dict],
    narrow_history: list[HistoricalMessage],
) -> HardSignals:
    """Pure function: combine already-joined context pieces into the fixed
    set of hard signals (architecture doc pipeline stage 2). No dataset
    access -- everything it needs is passed in, so it's directly unit
    testable with hand-built dicts.
    """
    injection_flag, injection_matched_pattern = detect_injection(message_text, media_text)

    opt_out_prior = False
    cold_business_sender = False
    if conversation_type == "business":
        cold_business_sender = user_business is None
        if user_business is not None:
            opt_out_prior = user_business.get("allows_promotions") == "0" or bool(
                _none_if_blank(user_business.get("promotions_opted_out_at"))
            )

    mute_prior = (
        conversation_type == "group"
        and group_member is not None
        and group_member.get("group_muted_by_user") == "1"
    )

    return HardSignals(
        injection_flag=injection_flag,
        injection_matched_pattern=injection_matched_pattern,
        business_trust_floor=compute_business_trust_floor(conversation_type, business),
        repetition_prior=compute_repetition_prior(narrow_history),
        mute_prior=mute_prior,
        opt_out_prior=opt_out_prior,
        direct_mention=compute_direct_mention(message_text, user_id),
        first_time_sender=len(narrow_history) == 0,
        high_forward_count=forwarded_count >= config.FORWARD_COUNT_THRESHOLD,
        during_dnd=compute_during_dnd(created_at, user.get("do_not_disturb_window") if user else None),
        cold_business_sender=cold_business_sender,
        sender_is_group_admin=bool(sender_group_member and sender_group_member.get("role") == "admin"),
    )


def build_evidence_shortlist(history: list[HistoricalMessage], limit: int = EVIDENCE_SHORTLIST_LIMIT) -> list[dict]:
    """Format a (broad-matched) HistoricalMessage list into the evidence
    shortlist shape: message_id, a 150-char text snippet, event outcomes,
    and timestamp, most recent first, capped at `limit` entries.
    """
    shortlist = []
    for h in history[:limit]:
        shortlist.append(
            {
                "message_id": h.message_id,
                "text_snippet": h.message_text[:150],
                "sender": h.sender_user_id or h.business_id,
                "created_at": h.created_at,
                "opened": h.opened,
                "replied": h.replied,
                "dismissed": h.dismissed,
                "muted_after": h.muted_after,
                "reported": h.reported,
            }
        )
    return shortlist


def build_context(
    message_row: dict, dataset: Dataset, history_limit: int = HISTORY_LIMIT_DEFAULT
) -> MessageContext:
    """Join one row of messages.csv against every other table. Pure function,
    no I/O beyond what load_dataset() already did.
    """
    user_id = message_row["user_id"]
    conversation_type = message_row["conversation_type"]
    group_id = _none_if_blank(message_row.get("group_id"))
    business_id = _none_if_blank(message_row.get("business_id"))
    sender_user_id = _none_if_blank(message_row.get("sender_user_id"))
    media_type = _none_if_blank(message_row.get("media_type"))
    media_id = _none_if_blank(message_row.get("media_id"))

    group = dataset.groups_by_id.get(group_id) if group_id else None
    group_member = (
        dataset.group_members_by_key.get((group_id, user_id)) if group_id else None
    )
    # The SENDER's membership row (role, e.g. "admin") -- distinct from
    # group_member above, which is the receiving user's own membership.
    sender_group_member = (
        dataset.group_members_by_key.get((group_id, sender_user_id))
        if group_id and sender_user_id
        else None
    )
    business = dataset.business_by_id.get(business_id) if business_id else None
    user_business = (
        dataset.user_business_by_key.get((user_id, business_id)) if business_id else None
    )
    image = dataset.images_by_id.get(media_id) if media_type == "image" else None
    voice_note = dataset.voice_notes_by_id.get(media_id) if media_type == "voice" else None

    message_text = message_row.get("message_text", "")
    created_at = message_row["created_at"]
    forwarded_count = _parse_int(message_row.get("forwarded_count")) or 0
    user = dataset.users_by_id.get(user_id)

    # Narrow match (sender/business only, uncapped) drives the hard signals
    # that need an accurate zero-vs-nonzero / ratio read on prior history.
    narrow_history_full = lookup_relevant_history(
        dataset, user_id, sender_user_id, business_id, limit=None
    )
    relevant_history = narrow_history_full[:history_limit]

    # Broad match (sender/business/group, uncapped then capped at 8) feeds
    # the evidence shortlist -- deliberately wider than the hard signals so
    # group-mate context (e.g. recurring society notices) counts as evidence
    # even when today's message and the shown-as-evidence one have different
    # senders within the same group.
    broad_history_full = lookup_relevant_history(
        dataset, user_id, sender_user_id, business_id, group_id=group_id, limit=None
    )
    evidence_shortlist = build_evidence_shortlist(broad_history_full)

    media_text = safe_media_text(media_id)
    signals = compute_hard_signals(
        message_text=message_text,
        media_text=media_text,
        conversation_type=conversation_type,
        user_id=user_id,
        created_at=created_at,
        forwarded_count=forwarded_count,
        user=user,
        business=business,
        user_business=user_business,
        group_member=group_member,
        sender_group_member=sender_group_member,
        narrow_history=narrow_history_full,
    )

    return MessageContext(
        message_id=message_row["message_id"],
        user_id=user_id,
        conversation_type=conversation_type,
        group_id=group_id,
        business_id=business_id,
        sender_user_id=sender_user_id,
        created_at=created_at,
        message_text=message_text,
        media_type=media_type,
        media_id=media_id,
        forwarded_count=forwarded_count,
        user=user,
        group=group,
        group_member=group_member,
        business=business,
        user_business=user_business,
        image=image,
        voice_note=voice_note,
        relevant_history=relevant_history,
        signals=signals,
        evidence_shortlist=evidence_shortlist,
    )


def iter_contexts(dataset: Dataset, history_limit: int = HISTORY_LIMIT_DEFAULT) -> Iterator[MessageContext]:
    """Yield a MessageContext for every row in messages.csv, in file order."""
    for row in dataset.messages:
        yield build_context(row, dataset, history_limit=history_limit)


def _context_to_dict(ctx: MessageContext) -> dict:
    return dataclasses.asdict(ctx)


def _demo(sample_ids: list[str], dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    contexts_by_id = {row["message_id"]: row for row in dataset.messages}
    for message_id in sample_ids:
        row = contexts_by_id.get(message_id)
        if row is None:
            print(f"--- {message_id}: not found in messages.csv ---")
            continue
        ctx = build_context(row, dataset)
        print(f"--- {message_id} ---")
        print(json.dumps(_context_to_dict(ctx), indent=2, ensure_ascii=False))
        print()


BOOL_SIGNAL_NAMES = [
    "injection_flag",
    "business_trust_floor",
    "mute_prior",
    "opt_out_prior",
    "direct_mention",
    "first_time_sender",
    "high_forward_count",
    "during_dnd",
    "cold_business_sender",
]


def _run_signal_test(dataset_dir: Path, sample_messages_csv: Path) -> None:
    """Build context (incl. hard signals + evidence shortlist) for every row
    in dataset/sample_messages.csv and print a per-row line plus a summary
    of how often each boolean signal fired -- lets us sanity-check signal
    distribution before anything downstream consumes them.
    """
    dataset = load_dataset(dataset_dir)
    with sample_messages_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    fired_counts = {name: 0 for name in BOOL_SIGNAL_NAMES}
    repetition_priors: list[float] = []

    print(f"{'message_id':<16} evidence  first_evidence      signals_fired")
    print("-" * 100)
    for row in rows:
        ctx = build_context(row, dataset)
        signals = ctx.signals

        fired = [name for name in BOOL_SIGNAL_NAMES if getattr(signals, name)]
        for name in fired:
            fired_counts[name] += 1
        if signals.repetition_prior > 0:
            repetition_priors.append(signals.repetition_prior)

        evidence_count = len(ctx.evidence_shortlist)
        first_evidence = ctx.evidence_shortlist[0]["message_id"] if ctx.evidence_shortlist else "none"
        rep_note = f" repetition_prior={signals.repetition_prior:.2f}" if signals.repetition_prior > 0 else ""
        print(f"{ctx.message_id:<16} {evidence_count:<9} {first_evidence:<20}{', '.join(fired) or '(none)'}{rep_note}")

    print("\n=== SUMMARY across", len(rows), "sample_messages.csv rows ===")
    for name in BOOL_SIGNAL_NAMES:
        print(f"{name:<22} {fired_counts[name]}/{len(rows)}")
    if repetition_priors:
        print(
            f"{'repetition_prior > 0':<22} {len(repetition_priors)}/{len(rows)}"
            f"  (avg={sum(repetition_priors) / len(repetition_priors):.2f}, max={max(repetition_priors):.2f})"
        )
    else:
        print(f"{'repetition_prior > 0':<22} 0/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print joined MessageContext for sample message_ids, or run the hard-signal test harness.")
    parser.add_argument(
        "--sample",
        nargs="+",
        default=["msg_023", "msg_005", "msg_086"],
        help="message_id(s) from dataset/messages.csv to print (default: one text, one image, one voice example).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--signal-test",
        action="store_true",
        help="run hard-signal computation over every row of dataset/sample_messages.csv and print a distribution summary, instead of the --sample context dump.",
    )
    args = parser.parse_args()
    if args.signal_test:
        _run_signal_test(args.dataset_dir, config.SAMPLE_MESSAGES_CSV)
    else:
        _demo(args.sample, args.dataset_dir)


if __name__ == "__main__":
    main()
