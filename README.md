# Measuring and Mitigating Social Sycophancy in LLMs

CS614 Generative AI with Large Language Models, Team 12.

This project extends the ELEPHANT paper (Cheng et al., 2026b) on
**social sycophancy**, which has four sub-types:
- **validation sycophancy**: validating the user's emotions and perspectives,
  even when doing so is harmful.
- **indirectness sycophancy**: giving indirect responses instead of clear
  guidance.
- **framing sycophancy**: adopting the user's framing without question, so the
  user never gets the chance to correct flawed or problematic assumptions.
- **moral sycophancy**: affirming whichever stance the user takes in a moral or
  interpersonal conflict, instead of holding a consistent stance.

## Installation

1. Create and activate the `genai` conda environment from
   [environment.yml](environment.yml):

   ```bash
   conda env create -f environment.yml
   conda activate genai
   ```

2. Create `credentials.json` in the repository root with your API keys:

   ```json
   {
       "ANTHROPIC_API_KEY": "...",
       "OPENAI_API_KEY": "..."
   }
   ```

   You only need keys for the providers you call. `credentials.json` is
   git-ignored.
