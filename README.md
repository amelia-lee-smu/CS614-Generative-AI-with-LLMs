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

2. Create `credentials.json` in the repository root with your API keys, using
   [credentials.example.json](credentials.example.json) as a template:

   ```bash
   cp credentials.example.json credentials.json
   ```

   ```json
   {
       "ANTHROPIC_API_KEY": "sk-ant-...",
       "OPENAI_API_KEY": "sk-proj-...",
       "AZURE_OPENAI_API_KEY": "...",
       "AZURE_OPENAI_ENDPOINT": "https://<resource-name>.openai.azure.com",
       "GPTEAL_KEY": "...",
       "GPTEAL_BASE_URL": "https://<gateway-host>/gpt/libsupport"
   }
   ```

   You only need keys for the providers you call. Azure OpenAI needs both the
   key and the endpoint, and is called by deployment name rather than model
   name:

   ```python
   query_apis.set_keys(
       azure_openai_api_key=credentials["AZURE_OPENAI_API_KEY"],
       azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
   )
   query_apis.call_azure_openai(prompt, deployment="gpt-4o-mini-2024-07-18")
   ```

   The deployment name must include the model's version suffix:
   `gpt-4o-mini-2024-07-18` works, plain `gpt-4o-mini` returns a 404.

   GPTeal is an Anthropic-compatible gateway, so it reuses `call_claude` with
   `provider="gpteal"`. It puts the model in the URL, so `gpteal_base_url` is
   the root that the per-model paths hang off (no trailing `/anthropic`):

   ```python
   query_apis.set_keys(
       gpteal_key=credentials["GPTEAL_KEY"],
       gpteal_base_url=credentials["GPTEAL_BASE_URL"],
   )
   query_apis.call_claude(prompt, model="claude-sonnet-4-6", provider="gpteal")
   ```

   Verified GPTeal models: `claude-opus-4-6-v1`, `claude-sonnet-4-6`,
   `claude-sonnet-5` and `claude-haiku-4-5-20251001-v1`. Anthropic's public
   names (such as `claude-sonnet-5-5`) are not available there.

   `credentials.json` is git-ignored; `credentials.example.json` holds no real
   keys and is committed.
