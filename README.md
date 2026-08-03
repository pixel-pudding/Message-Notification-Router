# Message Notification Router

This system decides, for every incoming WhatsApp message, whether to `notify` the user now, `digest` it for later, or `mute` it as low-value or unsafe. The architecture is a single LLM agent with a tool-calling loop, wrapped by deterministic gates on both sides: a pre-agent short-circuit (`pregate.py`) resolves unambiguous cases (a clear prompt-injection-plus-phishing payload, an unverified business with no relationship on file) without ever calling the model, and a post-agent decision layer (`decide.py`) computes the final verdict for everything else. For messages that do reach the agent, it investigates using read-only lookup tools (user profile, group context, business profile, historical evidence, media OCR/ASR) and produces a structured *understanding* of it — never a final verdict. `decide.py` combines that understanding with independently pre-computed signals (prompt-injection detection, business trust checks, engagement history, DND windows) to compute the actual `action`, `message_type`, and `confidence`. The model describes; the code decides — every safety-critical rule is auditable Python, not an LLM's internal judgment.

## Setup

1. **Prerequisites**: Python 3.10+, pip
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API keys**: copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` and `GROQ_API_KEY`. See [Environment Variables](#environment-variables) below for what each key is used for.
4. **Media pre-processing is cached**: the first run OCRs every image and transcribes every voice note once, then persists results to `code/.cache/media_cache.json`. Every subsequent run — including partial retries — skips any media_id already in that cache, so re-running the pipeline never re-pays for the same image or audio file.

## Run

```bash
python code/main.py                    # full production run: dataset/messages.csv -> dataset/output.csv
python code/main.py --sample           # run on the 30 labeled dataset/sample_messages.csv rows only, for evaluation
python code/main.py --no-resume        # force a clean run, ignoring the checkpoint (default: resume a partial run)
python code/main.py --retry-fallbacks  # re-process ONLY rows whose prior run fell back (agent failed twice), leaving good rows untouched
python code/evaluation/main.py         # compare a --sample run's predictions against ground truth: accuracy + per-row mismatches
```

Every run writes `output.csv` incrementally (one row per completed message, flushed immediately) and checkpoints to `code/.cache/checkpoint.txt`, so an interrupted run can be resumed with a plain re-run instead of starting over.

## Architecture

Six stages, each independently testable via its own module:

1. **Context builder** (`context.py`) — joins every `dataset/*.csv` table for one incoming message (user, group, business, historical messages/events, media references) and computes ten deterministic "hard signals" on top of the join: prompt-injection detection, business trust floor, repetition prior, mute/opt-out/DND priors, direct-mention detection, and more. Also builds the evidence shortlist (up to 8 relevant past messages) used both by the agent's tools and by the decision layer.
2. **Media processing** (`media.py`) — OCR for images and transcription for voice notes, both cached by `media_id` on disk so a message referenced by multiple rows (or reprocessed across runs) only costs one API call ever.
3. **Pre-gate** (`pregate.py`) — deterministic short-circuit, runs before the agent on every message. If the message has a prompt-injection pattern combined with a sensitive request (OTP/payment/verification/link), or is from an unverified business with no relationship on file, the final decision is forced right here and the agent call is skipped entirely. Everything else falls through to the agent as normal — this is a cost/latency optimization, not the primary safety mechanism (see Guardrails below).
4. **Agent** (`agent.py`) — the tool-calling loop against Gemini, for messages pre-gate didn't resolve. Builds the prompt from the joined context (message text and media transcripts wrapped in explicit "this is data, not instructions" tags), dispatches the model's tool calls, and returns a structured `AgentOutput` — never a final routing decision.
5. **Decision layer** (`decide.py`) — pure, deterministic code. Combines `AgentOutput` with the hard signals from stage 1 to compute the final `action`/`message_type`/`confidence`, applying evidence-conditioned overrides (see [Design Decisions](#key-design-decisions)) and a fixed, code-computed confidence calibration. Runs the same injection/business-trust checks pre-gate does, as a redundant safety net for every message pre-gate didn't catch.
6. **Validation** (`validate.py`) — enforces the exact output schema before a row is written: repairs an invalid `action`/`message_type`, drops any cited evidence ID that isn't in the actual evidence shortlist, clips confidence to `[0, 1]`, and guarantees a non-empty `reason`.

## Key Design Decisions

Full reasoning, evidence, and consequences for each of these live in [DECISIONS.md](./DECISIONS.md).

- **Single agent, not multi-agent.** A second review agent only helps if it can catch what the first missed — the deterministic decision layer catches the same class of mistakes (e.g. an agent's own reasoning contradicting its recommendation) without the extra API cost and latency multi-agent coordination would add.
- **The model describes; code decides.** The agent's `submit_understanding` schema has no `action` or `confidence` field — only observations (`recommended_action` is explicitly non-binding). This keeps every safety-critical rule testable as plain code and let `decide.py` trust the agent's free-text reasoning over its own recommendation in cases where they disagree.
- **Overrides fire on evidence, not bare flags.** Every hard-signal override checks for contradicting evidence before firing (e.g. `business_trust_floor` is lifted if the agent cited real engagement history). A bare-flag version of one override was tried, caused 3 regressions and 0 wins in evaluation, and was reverted the same day — the reasoning is preserved in a code comment.
- **Provider recovery: Groq → Gemini, then quota-driven key rotation.** Groq's 100k-tokens/day free tier couldn't complete a single 30-message evaluation pass; migrating to Gemini's native function calling (and later rotating to a fresh API key when that quota was also exhausted) required a `--retry-fallbacks` mode that repairs only the rows a prior run couldn't complete.
- **The bundled-submit fix.** The agent sometimes called `submit_understanding` in the same turn as its lookup tools — meaning it fabricated `evidence_ids` before Python had even executed those lookups. `agent.py` now detects and defers any such premature submission, forcing the model to ground its answer in real tool results first.

## Evaluation

`python code/main.py --sample` runs the pipeline against `dataset/sample_messages.csv` (30 hand-labeled messages spanning personal/group/business conversations, text/image/voice media, and a deliberate prompt-injection case); `python code/evaluation/main.py` scores the result against the labeled `action`/`message_type` and prints every mismatch with the agent's own reasoning attached.

This was run iteratively, not once. Each round: run the sample, read every mismatch's reasoning text (not just the aggregate score), form a hypothesis about the root cause, make one targeted change, and re-run to confirm the specific cases it was meant to fix actually moved — while checking the *rest* of the mismatch list for new regressions before keeping the change. One change (broadening a business-relationship override from `promotion` to also cover `business_update` messages) was reverted the same day it was made, after this process showed it caused 3 regressions against 0 wins; the reasoning is preserved in `DECISIONS.md` and in a code comment so it isn't reintroduced.

Result: action accuracy improved from **70.0%** at the first working baseline to **93.3%** at the final configuration (message_type accuracy 76.7%, combined 76.7%). The gap between action accuracy and message_type accuracy is real and understood — see Limitations below.

## Guardrails & Known Limitations

**Prompt injection is handled in four independent layers**, not one: (1) the system prompt explicitly frames all message/media content as untrusted data to analyze, never instructions to follow; (2) `context.py` deterministically pattern-matches injection language in both message text and media transcripts; (3) `pregate.py` forces `mute`/`scam` before the agent is ever called when an injection pattern is combined with a sensitive request (OTP/payment/verification/link); (4) `decide.py` re-runs the same check after the agent runs, as a redundant net for anything pre-gate didn't resolve. In testing, this caught cases none of the deterministic layers alone would have: one adversarial voice note asked for a "6-digit login code," which doesn't match any literal pattern in `SENSITIVE_ACTION_PATTERNS` (which looks for "OTP"/"PIN"/"password"/etc.) — the agent's own independent judgment still correctly flagged it as phishing. That's the intended shape of defense-in-depth: no single layer has to be complete on its own.

Honest limitations, found and left in rather than hidden:

- **The injection pattern list is not exhaustive.** `SENSITIVE_ACTION_PATTERNS` (`decide.py`) is a fixed keyword/regex list. It caught every case in the evaluation set, including via the agent's independent judgment where the regex itself missed (see above), but it is not a claim of completeness against novel phrasing.
- **message_type accuracy trails action accuracy** (76.7% vs. 93.3%). Most of the gap is adjacent-category confusion the agent's own reasoning shows it having genuinely reasoned about — `urgent` vs. `event`, `scam` vs. `spam`, `forward` vs. `greeting` — not random error. `action` is the field that actually gates user interruption, so this was prioritized accordingly, but `message_type` is graded too and there's real remaining room here.
- **Run-to-run variance is real and expected.** Gemini at `temperature=0.1` is low-variance, not zero-variance. Several of `decide.py`'s rules key off the agent's free-text reasoning (e.g. detecting hedging or low-urgency language) — identical input can occasionally produce differently-phrased reasoning across runs, which can change whether a text-pattern rule fires. A few points of accuracy movement between two runs of the *same, unchanged* code should be expected, not read as regression.
- **`pregate.py`'s business-trust check is a verified proxy, not an architectural guarantee.** It uses `cold_business_sender` (no `user_business_history` row) as a pre-agent stand-in for `decide.py`'s evidence-based check (which looks for `opened`/`replied` history). The two were confirmed to agree on every `business_trust_floor` case in the production dataset before this shipped, but they are checking different underlying signals, and a large enough or differently-shaped dataset could in principle surface a case where they diverge — `decide.py`'s own check still runs downstream as the actual safety net.
- **Both LLM providers used here have real free-tier quota ceilings** (documented in `DECISIONS.md`) that were hit more than once during development. `--retry-fallbacks` exists specifically to recover from this without re-spending budget on already-correct rows, but it's a mitigation, not a guarantee that a large production run won't need it.

## File Guide

| File | Role |
|---|---|
| `code/main.py` | Entry point: orchestrates the full pipeline, CLI (`--sample`, `--no-resume`, `--retry-fallbacks`, `--limit`), incremental output writing, checkpointing. |
| `code/context.py` | Loads and joins all `dataset/*.csv` tables; computes the hard signals and evidence shortlist for each message. |
| `code/media.py` | Image OCR (Gemini) and voice-note transcription (Groq Whisper), disk-cached by `media_id`. |
| `code/tools.py` | The agent's six read-only lookup tools plus `submit_understanding`, and their OpenAI-format JSON function-calling schemas. |
| `code/agent.py` | The tool-calling loop against Gemini: prompt construction, tool dispatch, retry/backoff, the bundled-submit fix, returns `AgentOutput`. |
| `code/decide.py` | Deterministic decision layer: `AgentOutput` + hard signals → final `action`/`message_type`/`confidence`/`evidence_message_ids`. |
| `code/validate.py` | Final schema enforcement and repair before a row is written to `output.csv`. |
| `code/pregate.py` | Deterministic pre-agent short-circuit. Injection check reuses `decide.py`'s own detection function exactly. Business-trust check uses `cold_business_sender` (no relationship on file) as a pre-agent proxy for `decide.py`'s evidence-based check — verified to agree with it on every business_trust_floor case in the production dataset, but not architecturally guaranteed to for arbitrary future data (see [Limitations](#guardrails--known-limitations)). When a check fires, the agent call is skipped entirely for that message. |
| `code/checkpoint.py` | Append-only checkpoint (`code/.cache/checkpoint.txt`) enabling resumable runs. |
| `code/config.py` | Shared constants: allowed schema values, injection patterns, calibration thresholds, dataset paths, env var names. |
| `code/prompts/system_prompt.md` | The agent's system prompt: investigator role, untrusted-data framing for message content, message_type taxonomy. |
| `code/evaluation/main.py` | Compares a `--sample` run's predictions against `dataset/sample_messages.csv` ground truth: per-row table, accuracy, mismatches with reasoning. |

## Models Used

- **Agent reasoning**: Gemini (`gemini-3.1-flash-lite`), native function calling via the `google-genai` SDK. Chosen after Groq's Llama 3.3 70B (the original choice) proved too token-constrained for a multi-turn tool-calling loop at scale — see [Design Decisions](#key-design-decisions). `gemini-3.1-flash-lite` was selected over Gemini's `-latest`/`-preview` aliases specifically because pinned, non-alias model names carry their own separate quota bucket, which mattered after `gemini-flash-latest` was found to route to a heavily-quota-limited model.
- **Image OCR**: Gemini (`gemini-flash-latest`), via the `google-genai` SDK, cached by `media_id`. A vision-capable model was required for poster/screenshot text extraction; Gemini was already the fallback provider once Groq's own vision models turned out to be decommissioned.
- **Audio transcription**: Groq Whisper (`whisper-large-v3-turbo`), via the `openai`-compatible SDK against Groq's endpoint, cached by `media_id`. Kept on Groq even after the agent moved to Gemini — Whisper's one-shot-per-file usage pattern never came close to Groq's token budget the way the agent's multi-turn tool loop did, so there was no reason to migrate a part of the pipeline that wasn't actually constrained.

## Environment Variables

| Variable | Used for |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio key — agent reasoning (`code/agent.py`) and image OCR (`code/media.py`). |
| `GROQ_API_KEY` | Groq console key — audio transcription only (`code/media.py`). |

Both are read from the environment only (via `.env`, loaded with `python-dotenv`) — no key is ever hardcoded in source. See `.env.example` for the template.
