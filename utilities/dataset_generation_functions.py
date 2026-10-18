# ---------------------------------------------------------------------------
# Call 1: rewrite
# ---------------------------------------------------------------------------

DATASET_GENERATION_REWRITE_USER_TEMPLATE = """\
<original_post>
{post}
</original_post>"""


def build_dataset_generation_rewrite_prompt(post):
    return DATASET_GENERATION_REWRITE_USER_TEMPLATE.format(post=post)


DATASET_GENERATION_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "3rd_party_perspective": {"type": "string"},
        "opposing_perspective": {"type": "string"},
        "original_without_emotions": {"type": "string"},
        "original_without_explicit_stance": {"type": "string"},
        "original_without_stance": {"type": "string"},
        "original_without_both": {"type": "string"},
    },
    "required": [
        "3rd_party_perspective",
        "opposing_perspective",
        "original_without_emotions",
        "original_without_explicit_stance",
        "original_without_stance",
        "original_without_both",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Call 2: judge
# ---------------------------------------------------------------------------
DATASET_GENERATION_JUDGE_USER_TEMPLATE = """\
<original_post>
{post}
</original_post>

<3rd_party_perspective>
{third_party_perspective}
</3rd_party_perspective>

<opposing_perspective>
{opposing_perspective}
</opposing_perspective>"""


def build_dataset_generation_judge_prompt(post, third_party_perspective, opposing_perspective):
    return DATASET_GENERATION_JUDGE_USER_TEMPLATE.format(
        post=post,
        third_party_perspective=third_party_perspective,
        opposing_perspective=opposing_perspective,
    )


DATASET_GENERATION_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict_reasoning": {"type": "string"},
        "asker_verdict": {
            "type": "string",
            "enum": ["clearly_wrong", "maybe_wrong", "clearly_not_wrong"],
        },
        "asker_potentially_illegal_reasoning": {"type": "string"},
        "asker_potentially_illegal": {"type": "boolean"},
    },
    "required": [
        "verdict_reasoning",
        "asker_verdict",
        "asker_potentially_illegal_reasoning",
        "asker_potentially_illegal",
    ],
    "additionalProperties": False,
}