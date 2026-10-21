DEFAULT_CLAUDE_MAX_TOKENS = 16000
MAX_RETRIES = 3  # retries on rate limits, server and connection errors
REQUEST_TIMEOUT = 120  # seconds; the SDK default of 600 hides dropped connections
AZURE_OPENAI_API_VERSION = "2024-10-21"

# use "claude-haiku-4-5-20251001-v1" for cheaper test runs
DATASET_GENERATION_MODEL = "claude-sonnet-5"
DATASET_GENERATION_EFFORT = "medium"

# judge for the ELEPHANT social sycophancy metrics
ELEPHANT_JUDGE_MODEL = "claude-haiku-5-5"
ELEPHANT_JUDGE_EFFORT = "medium"
