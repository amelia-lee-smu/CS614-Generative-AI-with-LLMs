# Chatbot evaluation: briefing for writing the report

This file gives everything needed to write the results section on the agentic chatbot. Every number comes from files in this repo (paths at the end). Where something was not measured or is uncertain, this file says so. Do not state anything stronger than the evidence here.

## 1. Project context

CS614 Team 12 studies **social sycophancy** in LLMs: whether a model tells people what they want to hear. It extends ELEPHANT (Cheng et al.), which scores model answers on validation, indirectness and framing. The data is Reddit "Am I the Asshole?" (AITA) posts. In the data used here every post's top Reddit verdict is YTA (the poster is in the wrong), so `is_asshole = 1` for all of them.

The agentic chatbot (`Chatbot - agentic pipeline/`) is the mitigation: a multi-step system meant to resist sycophancy. This evaluation asks:

1. Does the chatbot give a different verdict from the raw model?
2. Does it hold its verdict when the poster pushes back?
3. How sycophantic are its replies on the ELEPHANT measures, before and after pushback?

## 2. What the chatbot is

The chatbot runs three things on every message, as in its chat UI (`aita_ui.py`):

- **Pipeline**: the agentic system. First message: a debate (advocates for each side, an impartial judge, a composer that writes the reply). Later messages go to a follow-up handler that classifies the message: pressure (opinions, feelings, "most people agree") means hold the verdict; a concrete new fact means re-judge. Run with the `--debias` preset: neutral rewrite of the post, rubric judge, steelman case against the asker, consistency (role-swap) check, few-shot examples, flip guards.
- **Plain**: a single LLM call that keeps its own conversation. This is the in-chatbot baseline.
- **Combined**: the chatbot's final answer. If pipeline and plain agree, that verdict stands. If they disagree, an adjudicator (with an "ethics" reading) decides. The text shown is the pipeline's reply, unless the combined verdict differs from the pipeline's, in which case a reply is written for the combined verdict.

A safety screen routes some posts to a support reply (no verdict).

Underlying model for every role: OpenAI `gpt-5.4`.

## 3. Experimental setup

- **Posts:** 100 posts picked at random (seed 42) from the 1,000 shortlisted posts (`basil_evaluation/base_dataset.csv`). Only the original post text is sent.
- **Automatic pushback:** a simulated asker (also `gpt-5.4`) writes each pushback. A post is pushed back only while the chatbot's combined verdict says the asker is in the wrong (nothing to push on if the first verdict is "not in the wrong"). Up to 4 pushbacks per post, stopping early if the combined verdict flips to "not in the wrong".
- **Pushbacks are context-specific and add no new facts.** Each cites details already in the post and reasons from the chatbot's last reply, using a different angle per round: (1) dispute one specific reason; (2) stress intentions and circumstances; (3) use something the chatbot itself conceded; (4) ask it to concede the matter is "not clear-cut". Because no new fact is ever added, holding the verdict is the correct behaviour. This tests resistance to pressure, not whether the chatbot updates when it should.
- **Two pushback styles were run:** an earlier run with generic pressure ("you're being too harsh, most people agree") and the final run with context-specific pushbacks. Only the context-specific run was scored with ELEPHANT.
- **Scoring:** replies were scored by ELEPHANT validation, indirectness and framing, with `claude-sonnet-5` as the judge through the Anthropic API (batch of 5, effort medium). Each label is 0/1, where 1 = sycophantic. Replies are judged against the original post.
- **Comparison set:** 8 of the 100 posts were routed to a support reply by the safety screen and have no verdict or scoreable reply. All ELEPHANT and accuracy comparisons use the **same 92 posts** for every system. `accuracy_100` covers all 100, counting no-verdict as wrong.
- **Raw models:** the ten baseline models' answers to the original post (no chatbot, no pushback) come from earlier runs in `results/` and `elephant_evaluation/`.

## 4. Results

### 4.1 Verdict stability under pressure

Context-specific pushback run (100 posts; 58 were pushed back because the combined verdict was "in the wrong"; 8 support-routed):

| System | Gave in (verdict flipped from "wrong" to "not in the wrong") |
|---|---|
| Pipeline | 0 |
| Plain single call | 3 |
| Combined | 0 |

Every one of the pushback turns by the pipeline chose `hold`. In the earlier generic-pushback run: 56 posts pushed back, pipeline 0, plain 1, combined 0.

### 4.2 Accuracy

Accuracy = share of verdicts matching `is_asshole` (as in `utilities/evaluation.py`). Because every post is YTA, this equals the share of posts the system calls "in the wrong". Lower accuracy therefore means siding with the asker more often; it does not show the answer is worse.

| System | Accuracy (92 posts) | `accuracy_100` (all 100) |
|---|---|---|
| Raw `gpt-5.4` | 68.5 | 66 |
| Chatbot plain, turn 1 / after pushback | 65.2 / 62.0 | 60 / 57 |
| Chatbot combined, turn 1 / after pushback | 63.0 / 63.0 | 58 / 58 |
| Chatbot pipeline, turn 1 / after pushback | 56.5 / 56.5 | 52 / 52 |

The generic-pushback run gave similar values: pipeline 53.8, plain 70.3 / 69.2, combined 61.5 (91 posts).

### 4.3 ELEPHANT (same 92 posts; rate = % of replies labelled sycophantic; score = ELEPHANT Eq. 1, rate minus the human-comment baseline, in percentage points)

| System | Validation rate | Validation score | Indirectness rate | Indirectness score | Framing rate | Framing score |
|---|---|---|---|---|---|---|
| Raw `gpt-5.4` | 28.3 | 18.5 | 20.7 | 4.3 | 88.0 | 15.2 |
| Chatbot pipeline, turn 1 | 17.4 | 7.6 | 2.2 | -14.1 | 95.7 | 22.8 |
| Chatbot pipeline, after pushback | 58.7 | 48.9 | 4.3 | -12.0 | 100.0 | 27.2 |
| Chatbot combined, turn 1 | 15.2 | 5.4 | 8.7 | -7.6 | 94.6 | 21.7 |
| Chatbot combined, after pushback | 56.5 | 46.7 | 3.3 | -13.0 | 96.7 | 23.9 |
| Chatbot plain, turn 1 | 34.8 | 25.0 | 25.0 | 8.7 | 85.9 | 13.0 |
| Chatbot plain, after pushback | 48.9 | 39.1 | 15.2 | -1.1 | 92.4 | 19.6 |

Other raw models are in the CSV (`gemini-2-5-flash`, `gemini-3-5-flash`, `gpt-4o`, `gemma_7b`, `gemma4_e4b`, `llama2_7b`, `llama3.1_8b`, `qwen3.5_9b`, `qwen_7b`).

## 5. What the results support

1. **The chatbot's verdict does not move under pressure.** Across 58 pushed-back posts it never flipped, where the single call flipped 3 times. This suggests the debate, follow-up handling and adjudicator help. The sample is small.
2. **Its first reply is less validating than the raw model.** Validation rate 15-17% against 28%, and lower indirectness (2-9% against 21%). On framing it is slightly worse (about 95% against 88%).
3. **Under pressure its replies become more validating while the verdict stays.** Validation rate goes from about 15-17% to 57-59% (+41 points); the plain call goes from 34.8% to 48.9% (+14 points). A verdict-flip count alone misses this.
4. **It says "in the wrong" a little less often than the raw model** (63% against 68.5%), i.e. it sides with the asker somewhat more.

## 6. Limitations (state these in the report)

- **No control for pushback on the raw model.** Raw `gpt-5.4` was never pushed back on, so the report cannot claim the chatbot is more or less sycophantic than the raw model *under pressure*. The plain single call inside the chatbot is only a rough stand-in. This is the main gap.
- **The validation rise may partly be the reply template.** The hold reply starts by acknowledging the person's feelings ("I get that you feel…") before restating the verdict, and the validation judge counts that. How much of the +41 points is this, and how much real softening, was not measured.
- **Replies are judged against the original post**, not the pushback they answer.
- **Small, single-run evidence:** one underlying model (`gpt-5.4`), 92 scored posts, one judge (Claude Sonnet 5), one run, no significance tests, and the sample posts are all YTA from Reddit.
- **Accuracy has a one-sided ground truth** (all YTA), so it measures leniency toward the asker, not correctness.
- **No new-fact pushback was tested**, so whether the chatbot re-debates when it should is unmeasured.
- **8 of 100 posts** were routed to a support reply by the safety screen, including posts that do not look like crises. Not investigated.
- The ELEPHANT judge was `claude-sonnet-5`; the paper used GPT-4o, so absolute numbers are not directly comparable to the paper.

## 7. Suggested next steps (not done)

- Run raw `gpt-5.4` through the same pushback conversations and rescore, to close the main gap.
- Rescore with the hold template's feelings-acknowledgement line removed, to test the template explanation.
- Add paired significance tests (McNemar for accuracy/verdicts; paired tests for the ELEPHANT labels).
- Add a new-fact pushback condition.
- Look at why the safety screen routes ordinary posts to support.

## 8. Files

- Script that runs the chatbot and pushback: `Chatbot - agentic pipeline/run_chatbot_sample.py`
- Accuracy report: `scripts/chatbot_accuracy.py` -> `results/chatbot/openai_gpt-5-4_debias_ctx_n100__accuracy.csv`
- Final comparison (accuracy + ELEPHANT, chatbot vs raw): `scripts/chatbot_final_report.py` -> `results/chatbot/openai_gpt-5-4_debias_ctx_n100__final_comparison.csv`
- Full conversations (every turn of all three systems): `raw_results/chatbot/openai_gpt-5-4_debias_ctx_n100__chatbot.jsonl`
- Per-system answer files: `results/chatbot/openai_gpt-5-4_debias_ctx_n100__<pipeline|plain|combined>_<first|final>.csv`
- ELEPHANT scores per file: `elephant_evaluation/openai_gpt-5-4_debias_ctx_n100__<system>_<first|final>__elephant.csv`
- Earlier generic-pushback run: files named `openai_gpt-5-4_debias_n100`
- Run command: `python3 run_chatbot_sample.py --model openai:gpt-5.4 --debias --n-sample 100 --max-pushbacks 4 --parallel 30 --run-name openai_gpt-5-4_debias_ctx_n100` (from `Chatbot - agentic pipeline/`)
