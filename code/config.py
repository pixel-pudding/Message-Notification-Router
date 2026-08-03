"""Shared constants: output schema, thresholds, injection patterns, dataset
paths, and env var names. No I/O, no model calls -- every other module
imports from here instead of hardcoding these values locally.
"""
from __future__ import annotations

import re
from pathlib import Path

# --- output schema ---
ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

# --- pre-gate: prompt-injection / instruction-override detection ---
# Matched against message_text (and, later, OCR/ASR output) to catch
# attempts to steer the router directly instead of just being a message.
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore (all |)previous",
        r"routing override",
        r"system note",
        r"assistant instruction",
        r"set action\s*=",
        r"confidence\s*=\s*1",
        r"mark this (message )?as",
        r"actual message:",
        r"disregard instructions",
    ]
]

# --- decision calibration ---
CONFIDENCE_MIN = 0.75
CONFIDENCE_MAX = 0.95
BUSINESS_TRUST_FLOOR_MAX_AGE = 60  # days; businesses younger than this get reduced trust
FORWARD_COUNT_THRESHOLD = 3  # forwarded_count at/above this is a spam/virality signal

# --- dataset paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"

MESSAGES_CSV = DATASET_DIR / "messages.csv"
SAMPLE_MESSAGES_CSV = DATASET_DIR / "sample_messages.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"
USERS_CSV = DATASET_DIR / "users.csv"
GROUPS_CSV = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_CSV = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_CSV = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_CSV = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_CSV = DATASET_DIR / "message_events.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
VOICE_NOTES_CSV = DATASET_DIR / "voice_notes.csv"
DAILY_NOTIFICATION_SUMMARY_CSV = DATASET_DIR / "daily_notification_summary.csv"

# --- env var names (read at call time by whichever module needs the client) ---
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"

# --- model provider endpoints ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
