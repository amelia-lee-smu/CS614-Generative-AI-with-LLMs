import json
from pathlib import Path

# credentials.json sits in the project root (git-ignored).
CREDENTIALS_PATH = Path(__file__).parent.parent / "credentials.json"


def load_credentials(path=CREDENTIALS_PATH):
    """
    Load the API keys from credentials.json.
    """
    with open(path) as f:
        return json.load(f)
