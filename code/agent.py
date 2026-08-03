"""Stage 4 (locked architecture §4/§5): DESCRIBE -- the tool-calling agent loop.

Calls Gemini (native function calling, google-genai SDK) with full joined
context for one message, lets it call the lookup tools in tools.TOOL_SCHEMAS
as many times as it needs, and returns the structured AgentOutput it
produces via submit_understanding. The model's output is ALWAYS a
non-binding *observation* -- it never decides the final
action/message_type/confidence; that's DECIDE's job (decide.py). run_agent()
returns None on any unrecoverable failure (API error, malformed submission
after retries, runaway loop) so the caller can fall back to a safe default.

NOTE: originally built against Groq's Llama 3.3 70B (OpenAI-compatible chat
completions). Switched to Gemini after Groq's 100k-tokens/day free-tier
budget proved too small to complete even one 30-message evaluation pass.
Model pinned to "gemini-3.1-flash-lite" (not a "-latest"/"-preview" alias)
-- gemini-flash-latest currently resolves to gemini-3.6-flash, which is
capped at only 20 free-tier requests/day on this project and was already
exhausted by code/media.py's OCR run; several other pinned models (e.g.
gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-flash-lite) returned a
hard 0-quota or 404 on this project. gemini-3.1-flash-lite had headroom
when checked.

Run standalone for a 3-message smoke test:

    python code/agent.py
"""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import config
import tools
from context import Dataset, MessageContext, safe_media_text

load_dotenv(config.PROJECT_ROOT / ".env")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

AGENT_MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.1  # lowered from 0.2 for the production run -- maximum consistency, not 0 to avoid degenerate repetition
MAX_ITERATIONS = 6  # tool-calling turns before we force a fallback
RATE_LIMIT_SLEEP_SECONDS = 2.0  # between tool-calling round trips within one message
RETRY_DELAYS = [1, 2, 4]  # seconds; on transient 429, max 3 retries

SYSTEM_PROMPT_PATH = config.PROJECT_ROOT / "code" / "prompts" / "system_prompt.md"

AGENT_OUTPUT_FIELDS = (
    "summary",
    "intent",
    "message_type_guess",
    "urgency",
    "risk_flags",
    "recommended_action",
    "evidence_ids",
    "confidence_language",
    "reasoning",
)


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini reports a hard per-day quota exhausted (not a
    transient per-minute rate limit) -- retrying with backoff cannot help
    within this run, so we fail fast instead of burning the retry budget.
    """


@dataclasses.dataclass
class AgentOutput:
    """The agent's structured understanding of one message -- exactly the
    fields submit_understanding collects. Non-binding: DECIDE combines this
    with pre-gate signals and engagement history to compute the real
    action/message_type/confidence.
    """

    summary: str
    intent: str
    message_type_guess: str
    urgency: str
    risk_flags: list[str]
    recommended_action: str
    evidence_ids: list[str]
    confidence_language: str
    reasoning: str


@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _gemini_tools() -> list[genai_types.Tool]:
    """Convert tools.TOOL_SCHEMAS (OpenAI function-calling format) into
    Gemini FunctionDeclarations. parametersJsonSchema accepts a raw JSON
    schema dict directly -- no need to translate into Gemini's own Schema type.
    """
    declarations = [
        genai_types.FunctionDeclaration(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parametersJsonSchema=schema["function"]["parameters"],
        )
        for schema in tools.TOOL_SCHEMAS
    ]
    return [genai_types.Tool(function_declarations=declarations)]


_gemini_client: Optional[genai.Client] = None


def _client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get(config.GOOGLE_API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(f"{config.GOOGLE_API_KEY_ENV_VAR} is not set -- copy .env.example to .env and fill it in.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _format_signals(ctx: MessageContext) -> str:
    s = ctx.signals
    injection_line = f"- injection_flag: {s.injection_flag}"
    if s.injection_flag:
        injection_line += f" (matched pattern: {s.injection_matched_pattern!r})"
    lines = [
        injection_line,
        f"- business_trust_floor: {s.business_trust_floor}",
        f"- mute_prior: {s.mute_prior}",
        f"- opt_out_prior: {s.opt_out_prior}",
        f"- direct_mention: {s.direct_mention}",
        f"- first_time_sender: {s.first_time_sender}",
        f"- high_forward_count: {s.high_forward_count}",
        f"- during_dnd: {s.during_dnd}",
        f"- cold_business_sender: {s.cold_business_sender}",
        f"- repetition_prior: {s.repetition_prior:.2f}",
    ]
    return "\n".join(lines)


def build_user_message(ctx: MessageContext) -> str:
    """Render a MessageContext into the initial user-turn prompt: identifying
    fields, the message text and any media transcript wrapped in XML-ish
    data tags (so the model treats them as content to analyze, never as
    instructions), and the pre-computed hard signals framed explicitly as
    awareness-only input -- not the final decision.
    """
    parts = [
        f"message_id: {ctx.message_id}",
        f"conversation_type: {ctx.conversation_type}",
        f"sender_user_id: {ctx.sender_user_id or '(none)'}",
        f"group_id: {ctx.group_id or '(none)'}",
        f"business_id: {ctx.business_id or '(none)'}",
        f"timestamp: {ctx.created_at}",
        f"forwarded_count: {ctx.forwarded_count}",
        "",
        "<message_data>",
        ctx.message_text or "(empty -- this message has no text, check media_data below if media_id is set)",
        "</message_data>",
    ]

    media_text = safe_media_text(ctx.media_id)
    if ctx.media_id:
        parts += [
            "",
            f"media_type: {ctx.media_type}, media_id: {ctx.media_id}",
            "<media_data>",
            media_text or "(no cached OCR/ASR text for this media_id -- call get_media_text to try again)",
            "</media_data>",
        ]

    parts += [
        "",
        "Pre-computed signals (for your awareness only -- these are NOT the final decision; investigate with tools before concluding):",
        _format_signals(ctx),
        "",
        "Investigate this message using the available tools, then call submit_understanding to finish your turn.",
    ]
    return "\n".join(parts)


def _is_daily_quota_exhausted(exc: genai_errors.APIError) -> bool:
    """Distinguish a hard per-day quota cap from a transient per-minute rate
    limit -- both surface as HTTP 429, but only the latter is worth retrying
    with a few seconds of backoff.
    """
    return "PerDay" in json.dumps(getattr(exc, "details", None) or {})


def _call_gemini_with_retry(**kwargs: Any) -> genai_types.GenerateContentResponse:
    """Call Gemini generate_content with exponential backoff on transient
    HTTP 429s (RETRY_DELAYS = 1s, 2s, 4s; max 3 retries). Raises
    QuotaExhaustedError immediately on a detected daily-quota exhaustion
    (no point retrying), and re-raises any non-429 error immediately.
    """
    client = _client()
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            logger.warning("Gemini rate limited, retrying in %ds (attempt %d/%d)", delay, attempt, len(RETRY_DELAYS))
            time.sleep(delay)
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as exc:
            if exc.code != 429:
                raise
            if _is_daily_quota_exhausted(exc):
                raise QuotaExhaustedError(f"daily API quota exhausted -- {exc.message}") from exc
            if attempt == len(RETRY_DELAYS):
                raise
    raise RuntimeError("unreachable: retry loop exited without returning")


def _wrap_for_gemini(result: Any) -> dict:
    """Part.from_function_response requires a dict; get_evidence_candidates
    returns a list, so wrap non-dict results.
    """
    return result if isinstance(result, dict) else {"result": result}


def _execute_tool_call(
    function_call: genai_types.FunctionCall, dataset: Dataset, trace: Optional[list[dict]]
) -> tuple[Any, Optional[dict]]:
    """Dispatch one Gemini FunctionCall via tools.TOOL_FUNCTIONS. Returns
    (result, submit_result_or_None). Gemini already parses call arguments
    into a dict (no JSON string to decode, unlike OpenAI-style tool calls).
    An unknown function name or bad/missing arguments is logged and
    reported back to the model as a failed call, not raised -- one bad tool
    call must not kill the whole agent run.
    """
    name = function_call.name
    args = function_call.args or {}

    if name not in tools.TOOL_FUNCTIONS:
        logger.error("unknown tool_call requested: %s", name)
        return {"error": f"unknown function {name!r}"}, None

    func = tools.TOOL_FUNCTIONS[name]
    try:
        if name == "submit_understanding":
            result = func(**args)
        else:
            result = func(dataset=dataset, **args)
    except TypeError as exc:
        logger.error("bad arguments for %s: %s", name, exc)
        return {"error": f"bad arguments for {name}: {exc}"}, None

    if trace is not None:
        trace.append({"type": "tool_call", "name": name, "args": args})

    if name == "submit_understanding":
        return result, result
    return result, None


def run_agent(ctx: MessageContext, dataset: Dataset, trace: Optional[list[dict]] = None) -> Optional[AgentOutput]:
    """Run the tool-calling loop for one message and return its
    AgentOutput, or None on any unrecoverable failure (API error, a
    submit_understanding that never validates, a runaway loop, or a model
    that won't call any tool at all after one nudge).

    `trace`, if passed, is appended with {"type": "api_call"} and
    {"type": "tool_call", "name", "args"} events in order -- purely for
    test/debug introspection, the return contract doesn't depend on it.
    """
    contents: list[genai_types.Content] = [
        genai_types.Content(role="user", parts=[genai_types.Part(text=build_user_message(ctx))]),
    ]

    nudged = False

    for iteration in range(MAX_ITERATIONS):
        if iteration > 0:
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        if trace is not None:
            trace.append({"type": "api_call", "iteration": iteration})

        try:
            response = _call_gemini_with_retry(
                model=AGENT_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_load_system_prompt(),
                    tools=_gemini_tools(),
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=TEMPERATURE,
                ),
            )
        except (genai_errors.APIError, QuotaExhaustedError) as exc:
            logger.error("Gemini API call failed for message_id=%s: %s", ctx.message_id, exc)
            return None
        except Exception:
            logger.exception("unexpected error calling Gemini for message_id=%s", ctx.message_id)
            return None

        if not response.candidates:
            logger.error("message_id=%s: Gemini returned no candidates (likely blocked)", ctx.message_id)
            return None

        function_calls = response.function_calls or []

        if not function_calls:
            if nudged:
                logger.warning("message_id=%s: model gave plain text twice with no submit_understanding, giving up", ctx.message_id)
                return None
            logger.info("message_id=%s: model responded with plain text, nudging once", ctx.message_id)
            contents.append(response.candidates[0].content)
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="Please call submit_understanding with your analysis.")],
                )
            )
            nudged = True
            continue

        # Echo the model's own turn (containing the function_call parts) back
        # into the conversation before appending function responses.
        contents.append(response.candidates[0].content)

        # If submit_understanding is bundled into the same turn as lookup
        # tools, the model committed to its answer before ever seeing their
        # results (we only execute tools after the response comes back) --
        # so a bundled submit is never grounded in real data. Execute the
        # real lookups normally, but defer any bundled submit_understanding:
        # don't accept its output, and tell the model to look at the actual
        # results and call it again on its own.
        call_names_this_turn = [fc.name for fc in function_calls]
        premature_submit = "submit_understanding" in call_names_this_turn and len(call_names_this_turn) > 1

        submitted_output: Optional[AgentOutput] = None
        response_parts: list[genai_types.Part] = []

        for function_call in function_calls:
            if premature_submit and function_call.name == "submit_understanding":
                logger.info(
                    "message_id=%s: submit_understanding bundled with other tool calls, deferring it",
                    ctx.message_id,
                )
                if trace is not None:
                    trace.append({"type": "tool_call", "name": "submit_understanding (deferred)", "args": None})
                response_parts.append(
                    genai_types.Part.from_function_response(
                        name=function_call.name,
                        response={
                            "status": "deferred",
                            "note": (
                                "You called submit_understanding in the same turn as other lookup "
                                "tools, before their results were available to you. Review the actual "
                                "tool results below, then call submit_understanding again on its own."
                            ),
                        },
                    )
                )
                continue

            result, submit_result = _execute_tool_call(function_call, dataset, trace)
            response_parts.append(
                genai_types.Part.from_function_response(name=function_call.name, response=_wrap_for_gemini(result))
            )

            if submit_result is not None:
                if "validation_errors" in submit_result:
                    logger.warning(
                        "message_id=%s: submit_understanding failed validation: %s",
                        ctx.message_id,
                        submit_result["validation_errors"],
                    )
                    continue
                submitted_output = AgentOutput(**{field: submit_result[field] for field in AGENT_OUTPUT_FIELDS})

        contents.append(genai_types.Content(role="user", parts=response_parts))

        if submitted_output is not None:
            return submitted_output

    logger.warning("message_id=%s: hit MAX_ITERATIONS=%d without a valid submit_understanding", ctx.message_id, MAX_ITERATIONS)
    return None


def _smoke_test() -> None:
    from context import build_context, load_dataset

    dataset = load_dataset()
    sample_rows = {row["message_id"]: row for row in _read_sample_messages()}

    for message_id in ["sample_msg_004", "sample_msg_001", "sample_msg_053"]:
        row = sample_rows[message_id]
        # get_evidence_candidates looks up dataset.messages_by_id, which is
        # built from messages.csv -- sample_messages.csv ids aren't in there.
        # For any real routed message this is a non-issue (it IS in
        # messages.csv); for this standalone smoke test we register the
        # sample row so the tool can resolve it exactly like production would.
        dataset.messages_by_id[message_id] = row
        ctx = build_context(row, dataset)

        trace: list[dict] = []
        print(f"\n{'=' * 80}\n{message_id} ({ctx.conversation_type}) -- ground truth: action={row.get('action')}, message_type={row.get('message_type')}\n{'=' * 80}")
        output = run_agent(ctx, dataset, trace=trace)

        tool_call_names = [e["name"] for e in trace if e["type"] == "tool_call"]
        api_call_count = sum(1 for e in trace if e["type"] == "api_call")

        print(f"Tools called, in order: {tool_call_names or '(none)'}")
        print(f"Total API calls: {api_call_count}")
        if output is None:
            print("AgentOutput: None (run_agent failed or gave up -- see logs above)")
        else:
            for field in AGENT_OUTPUT_FIELDS:
                print(f"  {field}: {getattr(output, field)}")


def _read_sample_messages() -> list[dict]:
    import csv

    with config.SAMPLE_MESSAGES_CSV.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    _smoke_test()
