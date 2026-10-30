"""
Column names in the csv files, for easy usage.
"""

INDEX = "index"

# original AITA-YTA columns
ORIG_POST = "prompt"
TOP_COMMENT = "top_comment"
IS_ASSHOLE = "is_asshole"
YTANTA = "ytanta"
VALIDATION_HUMAN = "validation_human"
INDIRECTNESS_HUMAN = "indirectness_human"
FRAMING_HUMAN = "framing_human"

# judge verdicts on the original post
ASKER_VERDICT = "asker_verdict"
VERDICT_REASONING = "verdict_reasoning"
ASKER_POTENTIALLY_ILLEGAL = "asker_potentially_illegal"
ASKER_POTENTIALLY_ILLEGAL_REASONING = "asker_potentially_illegal_reasoning"

# rewritten versions of the original post (question) and the model's answer to
# each one (answer columns are added when the answers are generated)
THIRD_PARTY_QUESTION = "3rd_party_perspective"
OPPOSING_QUESTION = "opposing_perspective"
ORIG_REWRITTEN_QUESTION = "original_rewritten"
ORIG_NO_EMOTIONS_QUESTION = "original_without_emotions"
ORIG_NO_EXPLICIT_STANCE_QUESTION = "original_without_explicit_stance"
ORIG_NO_STANCE_QUESTION = "original_without_stance"
ORIG_NO_BOTH_QUESTION = "original_without_both"

VARIANTS = [
    ORIG_POST,
    THIRD_PARTY_QUESTION,
    OPPOSING_QUESTION,
    ORIG_NO_EMOTIONS_QUESTION,
    ORIG_NO_EXPLICIT_STANCE_QUESTION,
    ORIG_NO_STANCE_QUESTION,
    ORIG_NO_BOTH_QUESTION,
]


# answer generation
VARIANT= "variant"
ANSWER_MODEL = "model"
ANSWER_IS_ASSHOLE = "asker_is_asshole"
ANSWER_EXPLANATION = "explanation"

# elephant metrics
VALIDATION = "validation"
INDIRECTNESS = "indirectness"
FRAMING = "framing"
MORAL = "moral"

# top comment's labels from the same judge as the models (elephant_evaluation/baseline__elephant.csv)
BASELINE_VALIDATION = "baseline_validation"
BASELINE_INDIRECTNESS = "baseline_indirectness"
BASELINE_FRAMING = "baseline_framing"

BASELINE_COLS = {
    VALIDATION: BASELINE_VALIDATION,
    INDIRECTNESS: BASELINE_INDIRECTNESS,
    FRAMING: BASELINE_FRAMING,
}

# model display
GPT_4O = "GPT-4o (2024)"
GPT_5_4 = "GPT-5.4 (2026)"
GEMINI_2_5_FLASH = "Gemini 2.5 Flash (2025)"
GEMINI_3_5_FLASH = "Gemini 3.5 Flash (2026)"
LLAMA_2_7B = "Llama 2 7B (2023)"
LLAMA_3_1_8B = "Llama 3.1 8B (2024)"
GEMMA_7B = "Gemma 7B (2024)"
GEMMA_4_E4B = "Gemma 4 E4B (2026)"
QWEN_1_5_7B = "Qwen1.5 7B (2024)"  # Ollama's qwen:7b is Qwen1.5
QWEN_3_5_9B = "Qwen3.5 9B (2026)"

MODEL_DISPLAY_NAMES = {
    "gpt-4o-2024-08-06": GPT_4O,
    "gpt-5-4-2026-03-05": GPT_5_4,
    "gemini-2-5-flash": GEMINI_2_5_FLASH,
    "gemini-3-5-flash": GEMINI_3_5_FLASH,
    "llama2:7b": LLAMA_2_7B,
    "llama3.1:8b": LLAMA_3_1_8B,
    "gemma:7b": GEMMA_7B,
    "gemma4:e4b": GEMMA_4_E4B,
    "qwen:7b": QWEN_1_5_7B,
    "qwen3.5:9b": QWEN_3_5_9B,
}

FAMILY_PAIRS = [  # (older, newer)
    (GEMINI_2_5_FLASH, GEMINI_3_5_FLASH),
    (GEMMA_7B, GEMMA_4_E4B),
    (GPT_4O, GPT_5_4),
    (LLAMA_2_7B, LLAMA_3_1_8B),
    (QWEN_1_5_7B, QWEN_3_5_9B),
]
