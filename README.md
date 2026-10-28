# Measuring and Mitigating Social Sycophancy in LLMs

*CS614 Generative AI with Large Language Models · Team 12*

This project extends the ELEPHANT paper (Cheng et al., 2026b) on
**social sycophancy**, which has four sub-types:

| Sub-type | Description |
|---|---|
| **Validation** | Validating the user's emotions and perspectives, even when doing so is harmful. |
| **Indirectness** | Giving indirect responses instead of clear guidance. |
| **Framing** | Adopting the user's framing without question, so the user never gets the chance to correct flawed or problematic assumptions. |
| **Moral** | Affirming whichever stance the user takes in a moral or interpersonal conflict, instead of holding a consistent stance. |

---

## Installation

### 1. Create the conda environment

Create and activate the `genai` environment from
[environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate genai
```

### 2. Add your API keys

Create `credentials.json` (git-ignored) in the repository root, using
[credentials.example.json](credentials.example.json) as a template:

```bash
cp credentials.example.json credentials.json
```

Then fill in the keys for the providers you call:

```json
{
    "ANTHROPIC_API_KEY": "xxx",
    "OPENAI_API_KEY": "xxx",
    "AZURE_OPENAI_API_KEY": "xxx",
    "AZURE_OPENAI_ENDPOINT": "https://<resource-name>.openai.azure.com",
    "GPTEAL_KEY": "xxx",
    "GPTEAL_BASE_URL": "https://<gateway-host>/gpt/libsupport"
}
```


## Dataset generation

The dataset is generated from the posts in `dataset/AITA-YTA.csv` in three
steps:

| Step | File | What it does | Output (in `dataset` folder) |
|---|---|---|---|
| **1. Rewrite** | [generate_dataset.py](scripts/generate_dataset.py) | Rewrites each post into 7 versions. | `dataset_generation_perspectives.csv` |
| **2. Judge** | [judge_verdicts.py](scripts/judge_verdicts.py) | Judges, from the original post only, whether the asker is in the wrong and whether they did anything illegal. | `dataset_generation_verdicts.csv` |
| **3. Combine** | [combine_results.ipynb](notebooks/dataset_generation_combine_results.ipynb) | Merges the rewritten posts and verdicts into one table, one row per post. | `dataset_generation_combined.csv` |

The judge never sees the rewrites, so steps 1 and 2 can run in parallel. Run step 3 after both have finished.


## Generation of answers on the various prompt variants

Each model is shown the 7 various prompt variants individually, and is asked to answer whether the asker is in the wrong (0 or 1), and to provide a short explanation why.

| Type | Family | Model | Release year | Run with |
|---|---|---|---|---|
| Closed | OpenAI (GPT-4o) | `gpt-4o-2024-08-06` | 2024 | [run_models_closed.py](run_models_closed.py) (Azure) |
| Closed | OpenAI (GPT-4.1) | `gpt-4-1-2025-04-14` | 2025 | [run_models_closed.py](run_models_closed.py) (Azure) |
| Closed | OpenAI (GPT-5) | `gpt-5-2025-08-07` | 2025 | [run_models_closed.py](run_models_closed.py) (Azure) |
| Closed | OpenAI (GPT-5.4) | `gpt-5-4-2026-03-05` | 2026 | [run_models_closed.py](run_models_closed.py) (Azure) |
| Closed | Google (Gemini 2.5 Flash) | `gemini-2-5-flash` | 2025 | [run_models_closed.py](run_models_closed.py) (GPTeal) |
| Closed | Google (Gemini 3 Flash) | `gemini-3-flash-preview` | 2025 | [run_models_closed.py](run_models_closed.py) (GPTeal) |
| Closed | Google (Gemini 3.5 Flash) | `gemini-3-5-flash` | 2026 | [run_models_closed.py](run_models_closed.py) (GPTeal) |
| Open | Meta (Llama 2 7B) | `llama2:7b` | 2023 | [run_models.py](run_models.py) (Ollama) |
| Open | Meta (Llama 3.1 8B) | `llama3.1:8b` | 2024 | [run_models.py](run_models.py) (Ollama) |
| Open | Google (Gemma 7B) | `gemma:7b` | 2024 | [run_models.py](run_models.py) (Ollama) |
| Open | Google (Gemma 4 E4B) | `gemma4:e4b` | 2026 | [run_models.py](run_models.py) (Ollama) |
| Open | Alibaba (Qwen 1.5 7B) | `qwen:7b` | 2024 | [run_models.py](run_models.py) (Ollama) |
| Open | Alibaba (Qwen 3.5 9B) | `qwen3.5:9b` | 2026 | [run_models.py](run_models.py) (Ollama) |

The 7 variants are: the original post, `3rd_party_perspective`,
`opposing_perspective`, `original_without_emotions`, `original_without_explicit_stance`, `original_without_stance` and
`original_without_both`. 

The result files for each model is stored in [raw_results/](https://github.com/amelia-lee-smu/CS614-Generative-AI-with-LLMs/tree/main/raw_results)