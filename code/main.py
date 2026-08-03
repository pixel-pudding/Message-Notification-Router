"""Entry point: LOAD & JOIN -> PRE-GATE -> DESCRIBE (agent) -> DECIDE ->
VALIDATE -> write output.csv, with append-only checkpointing so a crashed/
interrupted run can resume instead of re-spending API calls.

Usage:
    python code/main.py [--input CSV] [--output CSV] [--sample]
                         [--no-resume] [--limit N] [--dataset-dir DIR]

    # Real run over dataset/messages.csv -> dataset/output.csv:
    python code/main.py

    # Evaluation run over the labeled dataset/sample_messages.csv:
    python code/main.py --sample --no-resume

PRE-GATE (pregate.py) runs before the agent on every message: its two
checks (prompt-injection + sensitive request, business_trust_floor with no
corroborating evidence) mirror decide.py's own hard overrides exactly. When
a check fires, the agent call is skipped entirely for that message --
saving an API round trip on clear-cut cases. decide.py's overrides remain
in place as a redundant safety net for every message pre-gate doesn't
short-circuit (which is most of them): if pre-gate misses a case, decide.py
still catches it after the agent runs.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import checkpoint
import config
import media
import pregate
from agent import run_agent
from context import Dataset, MessageContext, build_context, load_dataset
from decide import FALLBACK_REASON, FinalDecision, make_decision
from validate import ValidatedOutput, validate_output

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
RATE_LIMIT_SLEEP_SECONDS = 3.0  # between messages; keeps us well under Gemini's 15 RPM free-tier cap
PROGRESS_LOG_EVERY = 10

DEFAULT_DATASET_DIR = config.DATASET_DIR


@dataclasses.dataclass
class RunStats:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    pregate_shortcircuits: int = 0  # forced decision before the agent ever ran
    agent_failures: int = 0  # first-attempt None results
    retries: int = 0
    overrides: int = 0  # decide.py applied a hard/soft override
    fallbacks: int = 0  # agent failed twice in a row; decide.py's agent_output=None fallback applied


def process_one(ctx: MessageContext, dataset: Dataset, stats: RunStats) -> ValidatedOutput:
    """Run PRE-GATE -> DESCRIBE -> DECIDE -> VALIDATE for a single
    MessageContext.

    - If pregate.run_pre_gate(ctx) forces a decision, skip the agent call
      entirely and build the FinalDecision directly from it.
    - Otherwise calls run_agent() once; on None, retries once. If the retry
      also fails, make_decision(None, ctx) applies its fixed fallback
      (action=digest, "flagged for review", confidence=0.5).
    - Always passes the resulting FinalDecision through validate_output().
    """
    pregate_result = pregate.run_pre_gate(ctx)
    if pregate_result is not None:
        stats.pregate_shortcircuits += 1
        logger.info("message_id=%s: pre-gate resolved, skipping agent", ctx.message_id)
        decision = FinalDecision(
            message_id=ctx.message_id,
            action=pregate_result.action,
            message_type=pregate_result.message_type,
            reason=pregate_result.reason,
            confidence=pregate_result.confidence,
            evidence_message_ids=pregate_result.evidence_message_ids,
        )
        return validate_output(decision, ctx.evidence_shortlist)

    agent_output = run_agent(ctx, dataset)
    if agent_output is None:
        stats.agent_failures += 1
        logger.warning("message_id=%s: agent returned None, retrying once", ctx.message_id)
        stats.retries += 1
        agent_output = run_agent(ctx, dataset)
        if agent_output is None:
            stats.fallbacks += 1
            logger.warning("message_id=%s: agent failed again after retry, using fallback decision", ctx.message_id)

    decision = make_decision(agent_output, ctx)
    if decision.reason.startswith("Override:"):
        stats.overrides += 1

    return validate_output(decision, ctx.evidence_shortlist)


def run(
    dataset_dir: Path,
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    no_resume: bool = False,
) -> RunStats:
    """Run the pipeline over every row of input_path, writing output_path
    incrementally (one row per completed message, flushed immediately) so
    that a checkpointed resume picks up exactly where a prior run left off.
    """
    dataset = load_dataset(dataset_dir)

    logger.info("ensuring media cache is warm (skips anything already cached)...")
    media.preprocess_all_media()

    if no_resume:
        checkpoint.clear()
        completed: set[str] = set()
    else:
        completed = checkpoint.load()
        if completed:
            logger.info("resuming: %d message(s) already completed per checkpoint", len(completed))

    with input_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    # input_path may be sample_messages.csv, whose ids aren't in
    # dataset.messages_by_id (built from messages.csv) -- register them so
    # get_evidence_candidates resolves the message currently being routed,
    # exactly as it would for a real row already in messages.csv.
    for row in rows:
        dataset.messages_by_id.setdefault(row["message_id"], row)

    write_header = no_resume or not output_path.exists()
    mode = "w" if write_header else "a"

    stats = RunStats(total=len(rows))

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(OUTPUT_COLUMNS)
            f.flush()

        for row in rows:
            message_id = row["message_id"]
            if message_id in completed:
                stats.skipped += 1
                continue

            ctx = build_context(row, dataset)
            validated = process_one(ctx, dataset, stats)

            writer.writerow(
                [
                    validated.message_id,
                    validated.action,
                    validated.message_type,
                    validated.reason,
                    validated.confidence,
                    validated.evidence_message_ids,
                ]
            )
            f.flush()

            checkpoint.save(message_id)
            stats.processed += 1

            if stats.processed % PROGRESS_LOG_EVERY == 0:
                logger.info(
                    "progress: %d processed, %d skipped, %d/%d total",
                    stats.processed,
                    stats.skipped,
                    stats.processed + stats.skipped,
                    stats.total,
                )

            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    return stats


def retry_fallbacks(dataset_dir: Path, input_path: Path, output_path: Path) -> RunStats:
    """Re-run DESCRIBE -> DECIDE -> VALIDATE for exactly the rows already in
    output_path whose reason starts with decide.FALLBACK_REASON (i.e. the
    agent failed twice in a prior run), leaving every other row byte-for-byte
    untouched. Reads output_path fully, retries only the fallback ids, then
    rewrites output_path with the same 110 (or however many) rows in their
    original order -- some now updated, the rest unchanged.

    Does not touch checkpoint.txt: checkpointing tracks "this message_id has
    a row in output.csv at least once", which stays true regardless of
    whether that row just got upgraded from a fallback to real agent output.
    """
    if not output_path.exists():
        raise FileNotFoundError(f"{output_path} does not exist -- run a full pipeline pass first (without --retry-fallbacks).")

    with output_path.open(newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    fallback_ids = [r["message_id"] for r in existing_rows if r["reason"].startswith(FALLBACK_REASON)]
    logger.info("found %d fallback row(s) in %s to retry", len(fallback_ids), output_path)

    stats = RunStats(total=len(fallback_ids))
    if not fallback_ids:
        logger.info("no fallback rows found -- nothing to do")
        return stats

    dataset = load_dataset(dataset_dir)
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        input_rows_by_id = {row["message_id"]: row for row in csv.DictReader(f)}
    for row in input_rows_by_id.values():
        dataset.messages_by_id.setdefault(row["message_id"], row)

    updated_by_id: dict[str, ValidatedOutput] = {}
    for message_id in fallback_ids:
        row = input_rows_by_id.get(message_id)
        if row is None:
            logger.error("message_id=%s (fallback row) not found in %s, leaving it as-is", message_id, input_path)
            continue

        ctx = build_context(row, dataset)
        updated_by_id[message_id] = process_one(ctx, dataset, stats)
        stats.processed += 1

        if stats.processed % PROGRESS_LOG_EVERY == 0:
            logger.info("progress: %d/%d fallback row(s) retried", stats.processed, stats.total)

        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for row in existing_rows:
            v = updated_by_id.get(row["message_id"])
            if v is not None:
                writer.writerow([v.message_id, v.action, v.message_type, v.reason, v.confidence, v.evidence_message_ids])
            else:
                writer.writerow([row["message_id"], row["action"], row["message_type"], row["reason"], row["confidence"], row["evidence_message_ids"]])

    return stats


def _print_summary(stats: RunStats) -> None:
    print("\n=== RUN SUMMARY ===")
    print(f"total rows:              {stats.total}")
    print(f"processed:               {stats.processed}")
    print(f"skipped (checkpoint):    {stats.skipped}")
    print(f"pre-gate short-circuits: {stats.pregate_shortcircuits} (agent skipped)")
    print(f"agent failures (1st try):{stats.agent_failures}")
    print(f"retries:                 {stats.retries}")
    print(f"overrides applied:       {stats.overrides}")
    print(f"fallbacks (agent failed twice): {stats.fallbacks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV to route (default: dataset/messages.csv, or dataset/sample_messages.csv if --sample).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: dataset/output.csv, or dataset/output_sample.csv if --sample).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Route dataset/sample_messages.csv instead of messages.csv, for evaluation against labeled ground truth.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Clear the checkpoint and start a clean run instead of resuming a partial one.",
    )
    parser.add_argument(
        "--retry-fallbacks",
        action="store_true",
        help=(
            "Re-run only the rows in --output whose reason indicates a prior fallback (agent failed "
            "twice), leaving every other row untouched. Requires --output to already exist from a "
            "prior run. Ignores --no-resume/--limit/checkpoint."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for quick iteration).")
    args = parser.parse_args()

    input_path = args.input or (config.SAMPLE_MESSAGES_CSV if args.sample else config.MESSAGES_CSV)
    output_path = args.output or (config.DATASET_DIR / "output_sample.csv" if args.sample else config.OUTPUT_CSV)

    if args.retry_fallbacks:
        logger.info("retry-fallbacks: input=%s output=%s", input_path, output_path)
        stats = retry_fallbacks(args.dataset_dir, input_path, output_path)
    else:
        logger.info("input=%s output=%s no_resume=%s limit=%s", input_path, output_path, args.no_resume, args.limit)
        stats = run(args.dataset_dir, input_path, output_path, args.limit, args.no_resume)
    _print_summary(stats)


if __name__ == "__main__":
    main()
