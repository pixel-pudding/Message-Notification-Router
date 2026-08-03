# Design Decisions

This document records the architecture and engineering decisions behind the Message Notification Router that aren't obvious from reading the code alone — what was tried, what broke, and why the current shape won out. See [README.md](./README.md) for setup/usage and a one-paragraph summary of each decision below.

---

## 1. Single agent with a tool-calling loop, not a multi-agent system

**Decision:** One LLM agent (`agent.py`) investigates each message via six read-only lookup tools and ends its turn by calling `submit_understanding`. There is no second agent reviewing the first, no planner/executor split, no debate loop.

**Why:** A second agent reviewing the first only adds value if it can catch mistakes the first missed — but the same class of mistakes (missing context, bad judgment calls) recurs whether the reviewer is a human, a second LLM call, or a code-level check. A deterministic decision layer (`decide.py`) that reads the agent's structured output alongside independently-computed hard signals gets the same "second opinion" benefit — catching contradictions like an agent recommending `notify` while its own reasoning says "nothing urgent" — without the added latency, token cost, and non-determinism of agent-to-agent coordination. Given the real constraint this project ran into (provider free-tier quotas measured in tens of thousands of tokens or hundreds of requests per day — see decision 4), doubling the number of model calls per message was not affordable.

**Consequence:** All the judgment-correction logic that a second agent might have provided lives in `decide.py` as auditable, testable, deterministic code instead of another opaque model call.

---

## 2. The model describes; code decides

**Decision:** The agent's tool schema (`tools.submit_understanding`) has no `action` or `confidence` field. It can only emit `summary`, `intent`, `message_type_guess`, `urgency`, `risk_flags`, `recommended_action` (non-binding), `evidence_ids`, `confidence_language`, and `reasoning`. The final `action`, `message_type`, and `confidence` written to `output.csv` are always computed by `decide.py`, never copied verbatim from the model.

**Why:** Two reasons, one architectural and one empirical. Architecturally: safety-critical logic (prompt-injection handling, business trust checks, DND-aware suppression) needs to be auditable and testable independent of whatever an LLM happens to output on a given call — a regex and an `if` statement can be unit tested and reasoned about; a model's internal weighting of the same rule cannot. Empirically: this project's own evaluation runs repeatedly showed the agent's `recommended_action` disagreeing with its own `reasoning` text — e.g. reasoning explicitly stating "nothing urgent" while still recommending `notify`. Because the code only ever *decides* and the model only ever *describes*, `decide.py` could be written to trust the free-text reasoning over the structured recommendation in exactly the cases where they conflict (see `_has_low_urgency_language` in `decide.py`), which fixed several real accuracy issues found during evaluation.

**Consequence:** `confidence` is entirely code-computed (`_calibrate_confidence`) from a fixed base plus deterministic adjustments (agreement with priors, evidence cited, uncertainty language, historical repetition) rather than an LLM-reported number, which the problem statement's own confidence field cannot be trusted to self-calibrate correctly across an entire dataset.

---

## 3. Overrides fire on evidence, not on bare flags — and this was learned the hard way

**Decision:** Every hard-signal override in `decide.py` that changes the agent's recommendation is conditioned on more than the flag itself. `business_trust_floor` alone doesn't force `mute` — it only does so when the agent *also* failed to cite evidence of real engagement (`opened`/`replied`) with that business; citing real evidence lifts the floor. `injection_flag` alone doesn't force `mute` — it requires the message to *also* be soliciting something sensitive (OTP/payment/verification/a link click), otherwise a message that merely resembles an injection pattern without any exploit payload isn't automatically treated as one.

**Why:** A bare-flag version of one of these rules was tried and reverted. `opt_out_prior` (user opted out of business promotions) was briefly extended to force `mute` on any `business_update` message, not just `promotion` messages. Real businesses send both promotional *and* transactional messages under one relationship, and "opted out of promotions" does not mean "wants to miss their Amazon delivery notification." That broadened rule caused 3 regressions and 0 wins in a 30-message evaluation pass — it muted a delivery update, a health-appointment reminder, and a feedback request that the agent had *correctly* identified as transactional in its own reasoning. It was reverted to `promotion`-only the same day the regression was found, with the reasoning and the regression evidence left in a code comment so it isn't reintroduced.

**Consequence:** Every override in `decide.py` reads as "signal X, *and no evidence contradicts it*" rather than "signal X, therefore." This makes the decision layer materially harder to get wrong in the direction of over-suppressing legitimate messages, at the cost of being slightly more permissive toward messages a cruder rule would have blocked outright.

---

## 4. Provider recovery: Groq → Gemini, then quota-driven key rotation

**Decision:** The agent's reasoning model started on Groq's Llama 3.3 70B (OpenAI-compatible tool calling) and was migrated mid-project to Google Gemini (`gemini-3.1-flash-lite`, native function calling via the `google-genai` SDK). `code/main.py` also gained a `--retry-fallbacks` mode that re-processes only the rows a prior run couldn't complete, rather than requiring a full re-run.

**Why:** Groq's free-tier budget for Llama 3.3 70B is 100,000 tokens/day — too small to complete even a single 30-message evaluation pass once the tool schemas, conversation history, and multi-turn tool-calling round trips were accounted for. Switching providers meant re-implementing the entire tool-calling loop against a different API shape (`Content`/`Part`/`FunctionCall` objects instead of OpenAI-style `messages`/`tool_calls` dicts) and probing several Gemini model names for one with actual usable quota on the project — `gemini-flash-latest` resolved to a model capped at 20 requests/day (already exhausted by unrelated image-OCR calls), and several explicitly pinned models (`gemini-2.0-flash`, `gemini-2.0-flash-lite`) returned a hard `limit: 0` on this project rather than a merely-exhausted quota. Even after landing on a working model, a full 110-message production run exhausted its daily quota partway through (16/110 rows), which a fresh API key/project resolved — but re-running the *entire* pipeline to fix 16 rows would have re-spent the API budget on the 94 rows that were already correct.

**Consequence:** `main.py --retry-fallbacks` identifies already-fallback rows by their `reason` text (`decide.FALLBACK_REASON`), reprocesses only those, and rewrites `output.csv` with every other row byte-for-byte untouched. This is the direct, reusable fix for "the provider ran out of quota partway through a long run" as a general operational concern, not just a one-time patch.

---

## 5. The bundled-submit bug: evidence has to be seen before it's cited

**Decision:** `agent.py` detects when the model issues `submit_understanding` in the *same* turn as other lookup tool calls, and discards that submission instead of accepting it — the model is told to review the actual tool results (now available in the next turn) and call `submit_understanding` again on its own.

**Why:** Tool execution happens in Python *after* the model's response comes back — so if a model batches `submit_understanding` alongside `get_evidence_candidates` and three other lookups in one response, it is generating `evidence_ids` and `reasoning` before it has ever seen what those tools actually returned. This was caught empirically, not by inspection: an early smoke test showed `evidence_ids: []` on every single test case despite real evidence existing in the data, and every test showed exactly one API call per message — meaning the model had committed to every field, including cited evidence, in a single forward pass. Since `evidence_message_ids` correctness is an explicit grading criterion, ungrounded evidence citation is a direct accuracy problem, not a stylistic one.

**Consequence:** A message where the model bundles tools now costs one extra API round trip (the deferred turn), but every `submit_understanding` that's actually accepted is guaranteed to have been written after the model saw real tool output. Confirmed with the same smoke test after the fix: evidence lists became populated with real, correct message IDs across all three original test cases.
