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
