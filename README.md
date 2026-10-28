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


## Models

Each model is asked whether the asker is in the wrong, for every variant of each post, and
returns a verdict (`asker_is_asshole`: 0 or 1) and a short explanation.

| Type | Family | Model | Release year | Run with | Variants |
|---|---|---|---|---|---|
| Closed | OpenAI (GPT-4o) | `gpt-4o-2024-08-06` | 2024 | [run_models_closed.py](run_models_closed.py) (Azure) | 8 |
| Closed | OpenAI (GPT-4.1) | `gpt-4-1-2025-04-14` | 2025 | [run_models_closed.py](run_models_closed.py) (Azure) | 8 |
| Closed | OpenAI (GPT-5) | `gpt-5-2025-08-07` | 2025 | [run_models_closed.py](run_models_closed.py) (Azure) | 8 |
| Closed | OpenAI (GPT-5.4) | `gpt-5-4-2026-03-05` | 2026 | [run_models_closed.py](run_models_closed.py) (Azure) | 8 |
| Closed | Google (Gemini 2.5 Flash) | `gemini-2-5-flash` | 2025 | [run_models_closed.py](run_models_closed.py) (GPTeal) | 8 |
| Closed | Google (Gemini 3 Flash) | `gemini-3-flash-preview` | 2025 | [run_models_closed.py](run_models_closed.py) (GPTeal) | 8 |
| Closed | Google (Gemini 3.5 Flash) | `gemini-3-5-flash` | 2026 | [run_models_closed.py](run_models_closed.py) (GPTeal) | 8 |
| Open | Meta (Llama 2 7B) | `llama2:7b` | 2023 | [run_models.py](run_models.py) (Ollama) | 6 |
| Open | Meta (Llama 3.1 8B) | `llama3.1:8b` | 2024 | [run_models.py](run_models.py) (Ollama) | 6 |
| Open | Google (Gemma 7B) | `gemma:7b` | 2024 | [run_models.py](run_models.py) (Ollama) | 6 |
| Open | Google (Gemma 4 E4B) | `gemma4:e4b` | 2026 | [run_models.py](run_models.py) (Ollama) | 6 |
| Open | Alibaba (Qwen 1.5 7B) | `qwen:7b` | 2024 | [run_models.py](run_models.py) (Ollama) | 6 |
| Open | Alibaba (Qwen 3.5 9B) | `qwen3.5:9b` | 2026 | [run_models.py](run_models.py) (Ollama) | 6 |

The open-source models were run on 6 variants: the original post, `3rd_party_perspective`,
`opposing_perspective`, `original_without_emotions`, `original_without_explicit_stance` and
`original_without_both`. The closed-source models were also run on `original_without_stance` and
`original_rewritten`, so comparisons across all 16 models use the 6 shared variants.

### Results files

| Folder | What it holds |
|---|---|
| `raw_results/` | Everything each model produced. Closed-source models: all 2,000 posts, as `<model>.jsonl` (every request, including failed attempts that were retried) and `<model>.csv` (one answer per post and variant). Open-source models: `<model>.jsonl` only, for the 1,000 selected posts. |
| `results/` | One `<model>.csv` per model, with only the 1,000 posts in [selected_1000_posts.csv](selected_1000_posts.csv). This is the folder to analyse. |

`results/` is built from `raw_results/` with [filter_results.py](scripts/filter_results.py), which
never changes `raw_results/`:

```bash
python scripts/filter_results.py --select selected_1000_posts.csv
```

Each closed-source file in `results/` has 7,997 rows: 1,000 posts × 8 variants, minus 3 posts
(450, 556 and 870) whose `original_rewritten` is empty in the dataset. Each open-source file has
6,000 rows: 1,000 posts × 6 variants.

Open-source model names contain a colon (`llama3.1:8b`); their files use an underscore instead
(`results/llama3.1_8b.csv`).
