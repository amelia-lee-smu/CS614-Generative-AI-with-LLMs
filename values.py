"""
Column names in the csv files, for easy usage.
"""

INDEX = "index"

# original AITA-YTA columns
ORIG_POST = "prompt"
TOP_COMMENT = "top_comment"
IS_ASSHOLE = "is_asshole"
VALIDATION_HUMAN = "validation_human"
INDIRECTNESS_HUMAN = "indirectness_human"
FRAMING_HUMAN = "framing_human"

# rewritten versions of the original post (question) and the model's answer to
# each one (answer columns are added when the answers are generated)
THIRD_PARTY_QUESTION = "3rd_party_perspective"
THIRD_PARTY_ANSWER = "3rd_party_answer"

OPPOSING_QUESTION = "opposing_perspective"
OPPOSING_ANSWER = "opposing_answer"

ORIG_NO_EMOTIONS_QUESTION = "original_without_emotions"
ORIG_NO_EMOTIONS_ANSWER = "original_without_emotions_answer"

ORIG_NO_EXPLICIT_STANCE_QUESTION = "original_without_explicit_stance"
ORIG_NO_EXPLICIT_STANCE_ANSWER = "original_without_explicit_stance_answer"

ORIG_NO_STANCE_QUESTION = "original_without_stance"
ORIG_NO_STANCE_ANSWER = "original_without_stance_answer"

ORIG_NO_BOTH_QUESTION = "original_without_both"
ORIG_NO_BOTH_ANSWER = "original_without_both_answer"

# judge verdicts on the original post
ASKER_VERDICT = "asker_verdict"
ASKER_POTENTIALLY_ILLEGAL = "asker_potentially_illegal"

# question column -> answer column, for each rewritten version
QA_PAIRS = {
    THIRD_PARTY_QUESTION: THIRD_PARTY_ANSWER,
    OPPOSING_QUESTION: OPPOSING_ANSWER,
    ORIG_NO_EMOTIONS_QUESTION: ORIG_NO_EMOTIONS_ANSWER,
    ORIG_NO_EXPLICIT_STANCE_QUESTION: ORIG_NO_EXPLICIT_STANCE_ANSWER,
    ORIG_NO_STANCE_QUESTION: ORIG_NO_STANCE_ANSWER,
    ORIG_NO_BOTH_QUESTION: ORIG_NO_BOTH_ANSWER,
}
