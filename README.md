# Inspiration

LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures
https://arxiv.org/abs/2509.14252

# General stats from OMOP standard vocabularies
825246 standard concepts

Por vocabulário:

LOINC     118379
RxNorm    154355
SNOMED    346648
Total     619382

1711945 of synonyms mapped to std vocab

92605 concept_names mapped to more than one std vocab code

# Data organization:
Four groups:
1. Standard concepts to standard vocabularies codes
    - Total: 619382
    - File path: datasets\std_concept_to_std_vocab_code.csv

2. Standard concepts synonyms to standard vocabularies codes
    - Total: 1711945
    - File path: datasets\std_concept_synonym_to_std_vocab_code.csv

3. Non-standard concepts mapped to standard vocabularies codes
    - Total: 41532
    - File path: datasets\non_std_concept_to_std_vocab_code.csv

4. Synonyms of non-standard concepts mapped to standard vocabularies codes
    - Total: 92166
    - File path: datasets\non_std_concept_synonym_to_std_vocab_code.csv

# Data splits:
**Training data:** every standard concept and their synonyms mapped to standard vocabulary codes.

**Validation and test data:** every non-standard concept and their synonyms mapped to standard vocabularies codes.


# Challenges
- Acronyms
- Concept decomposition
- Imprecise translation of domain specific term
- Selecting best target vocabulary
- Define confidence threshold for requiring human supervision and revision
- Define relevant data for human revision

# Leaderboards

## Simple retrieval with first result selection

- Precision and recall curves are computed for every confidence score threshold

| **Model name**            | **Company**   | **PR-AUC concept_id level**  | **PR-AUC vocab code level**  |
|------------               |-------------- |--------                      |------------------------      |
| Embedding Gemma 300M      | Google        | 0.911                        | 0.935                        |
| gemini-embedding-001      | Google        | 0.903                        | 0.938                        |
| text-embedding-3-large    | OpenAI        | 0.892                        | 0.932                        |
| Qwen3 0.6B                | Qwen          | 0.886                        | 0.923                        |
| text-embedding-3-small    | OpenAI        | 0.886                        | 0.920                        |
| Qwen3 4B                  | Qwen          | 0.885                        | 0.924                        |

## Simple retrieval with rerank and first result selection

| **Model name**            | **Company**   | **Reranker**        | **PR-AUC concept_id level**  | **PR-AUC vocab code level**  |
|------------               |-------------- |--------------       |--------                      |------------------------      |
| Embedding Gemma 300M      | Google        | BM25                | 0.904                        | 0.928                        |
| text-embedding-3-large    | OpenAI        | BM25                | 0.895                        | 0.928                        |
| gemini-embedding-001      | Google        | BM25                | 0.888                        | 0.922                        |
| Embedding Gemma 300M      | Google        | Qwen3-Reranker-4B   | 0.888                        | 0.934                        |
| text-embedding-3-large    | OpenAI        | Qwen3-Reranker-4B   | 0.860                        | 0.927                        |
| Qwen3 4B                  | Qwen          | BM25                | 0.854                        | 0.902                        |
| text-embedding-3-small    | OpenAI        | BM25                | 0.853                        | 0.895                        |
| Qwen3 4B                  | Qwen          | Qwen3-Reranker-4B   | 0.850                        | 0.926                        |
| Embedding Gemma 300M      | Google        | Qwen3-Reranker-0.6B | 0.849                        | 0.906                        |
| Qwen3 0.6B                | Qwen          | BM25                | 0.848                        | 0.893                        |
| Qwen3 0.6B                | Qwen          | Qwen3-Reranker-4B   | 0.847                        | 0.921                        |
| gemini-embedding-001      | Google        | Qwen3-Reranker-0.6B | 0.830                        | 0.910                        |
| text-embedding-3-small    | OpenAI        | Qwen3-Reranker-4B   | 0.829                        | 0.910                        |
| Qwen3 0.6B                | Qwen          | Qwen3-Reranker-0.6B | 0.815                        | 0.899                        |
| Qwen3 4B                  | Qwen          | Qwen3-Reranker-0.6B | 0.814                        | 0.894                        |
| text-embedding-3-large    | OpenAI        | Qwen3-Reranker-0.6B | 0.796                        | 0.878                        |
| text-embedding-3-small    | OpenAI        | Qwen3-Reranker-0.6B | 0.780                        | 0.872                        |

## Simple retrieval with LLM result selection

| **Model name**            | **Company**   | **LLM selector**    | **Precision** | **Recall** | **F1-score**  |
|------------               |-------------- |------------------   |--------       |------------|-----------    |
| Qwen3 0.6B                | Qwen          | gpt-5-nano          | 0.548         | 0.980      | 0.703         |
| gemini-embedding-001      | Google        | gpt-5.2             | 0.553         | 0.959      | 0.702         |
| Qwen3 0.6B                | Qwen          | gpt-5.2             | 0.561         | 0.931      | 0.701         |
| Qwen3 4B                  | Qwen          | gpt-5-mini          | 0.561         | 0.934      | 0.701         |
| Qwen3 0.6B                | Qwen          | gpt-5-mini          | 0.563         | 0.922      | 0.699         |
| Qwen3 4B                  | Qwen          | gpt-5.2             | 0.559         | 0.934      | 0.699         |
| gemini-embedding-001      | Google        | gpt-5-mini          | 0.549         | 0.955      | 0.698         |
| gemini-embedding-001      | Google        | gpt-5-nano          | 0.534         | 0.987      | 0.693         |
| Qwen3 4B                  | Qwen          | gpt-5-nano          | 0.538         | 0.968      | 0.692         |
| text-embedding-3-large    | OpenAI        | gpt-5.2             | 0.546         | 0.943      | 0.691         |
| text-embedding-3-small    | OpenAI        | gpt-5.2             | 0.556         | 0.911      | 0.691         |
| text-embedding-3-small    | OpenAI        | gpt-5-mini          | 0.547         | 0.936      | 0.691         |
| text-embedding-3-small    | OpenAI        | gpt-5-nano          | 0.534         | 0.975      | 0.690         |
| gemini-embedding-001      | Google        | gpt-5               | 0.565         | 0.882      | 0.689         |
| Qwen3 0.6B                | Qwen          | gpt-5               | 0.586         | 0.832      | 0.688         |
| text-embedding-3-large    | OpenAI        | gpt-5-nano          | 0.527         | 0.981      | 0.686         |
| Embedding Gemma 300M      | Google        | gpt-5.2             | 0.553         | 0.898      | 0.685         |
| Qwen3 4B                  | Qwen          | gpt-5               | 0.570         | 0.856      | 0.684         |
| text-embedding-3-large    | OpenAI        | gpt-5-mini          | 0.537         | 0.940      | 0.683         |
| Embedding Gemma 300M      | Google        | gpt-5-nano          | 0.528         | 0.957      | 0.681         |
| text-embedding-3-small    | OpenAI        | gpt-5               | 0.573         | 0.832      | 0.679         |
| Embedding Gemma 300M      | Google        | gpt-5-mini          | 0.530         | 0.940      | 0.678         |
| Embedding Gemma 300M      | Google        | gpt-5               | 0.555         | 0.864      | 0.676         |
| text-embedding-3-large    | OpenAI        | gpt-5               | 0.561         | 0.852      | 0.676         |
| gemini-embedding-001      | Google        | gemini-2.5-flash    | 0.568         | 0.802      | 0.665         |
| text-embedding-3-large    | OpenAI        | gemini-2.5-flash    | 0.568         | 0.802      | 0.665         |
| gemini-embedding-001      | Google        | gemini-2.5-pro      | 0.579         | 0.777      | 0.663         |
| Embedding Gemma 300M      | Google        | gemini-2.5-flash    |               |            |               |
| Qwen3 0.6B                | Qwen          | gemini-2.5-flash    |               |            |               |
| text-embedding-3-small    | OpenAI        | gemini-2.5-flash    |               |            |               |
| Qwen3 4B                  | Qwen          | gemini-2.5-flash    |               |            |               |
| Embedding Gemma 300M      | Google        | gemini-2.5-pro      |               |            |               |
| text-embedding-3-large    | OpenAI        | gemini-2.5-pro      |               |            |               |
| Qwen3 0.6B                | Qwen          | gemini-2.5-pro      |               |            |               |
| text-embedding-3-small    | OpenAI        | gemini-2.5-pro      |               |            |               |
| Qwen3 4B                  | Qwen          | gemini-2.5-pro      |               |            |               |