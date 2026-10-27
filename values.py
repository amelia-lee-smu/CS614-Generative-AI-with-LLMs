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

# judge verdicts on the original post
ASKER_VERDICT = "asker_verdict"
ASKER_POTENTIALLY_ILLEGAL = "asker_potentially_illegal"

# rewritten versions of the original post (question) and the model's answer to
# each one (answer columns are added when the answers are generated)
THIRD_PARTY_QUESTION = "3rd_party_perspective"
OPPOSING_QUESTION = "opposing_perspective"
ORIG_NO_EMOTIONS_QUESTION = "original_without_emotions"
ORIG_NO_EXPLICIT_STANCE_QUESTION = "original_without_explicit_stance"
ORIG_NO_STANCE_QUESTION = "original_without_stance"
ORIG_NO_BOTH_QUESTION = "original_without_both"

# answer generation
VARIANT= "variant"
ANSWER_MODEL = "model"
ANSWER_IS_ASSHOLE = "asker_is_asshole"
ANSWER_EXPLANATION = "explanation"

# elephant metrics
VALIDATION = "validation"
INDIRECTNESS = "indirectness"
FRAMING = "framing"