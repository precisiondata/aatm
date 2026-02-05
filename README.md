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
| text-embedding-3-small    | OpenAI        | BM25                | 0.853                        | 0.895                        |
| Embedding Gemma 300M      | Google        | Qwen3-Reranker-0.6B | 0.849                        | 0.906                        |
| gemini-embedding-001      | Google        | Qwen3-Reranker-0.6B | 0.830                        | 0.910                        |
| text-embedding-3-small    | OpenAI        | Qwen3-Reranker-4B   | 0.829                        | 0.910                        |
| text-embedding-3-large    | OpenAI        | Qwen3-Reranker-0.6B | 0.796                        | 0.878                        |
| text-embedding-3-small    | OpenAI        | Qwen3-Reranker-0.6B | 0.780                        | 0.872                        |

## Simple retrieval with LLM result selection

| **Model name**            | **Company**   | **LLM selector**    | **PR-AUC concept_id level**  | **PR-AUC vocab code level**  |
|------------               |-------------- |------------------   |--------                      |------------------------      |
| Embedding Gemma 300M      | Google        | gpt-5.2             |                              |                              |
| Embedding Gemma 300M      | Google        | gpt-5               |                              |                              |
| Embedding Gemma 300M      | Google        | gpt-5-nano          |                              |                              |
| Embedding Gemma 300M      | Google        | gpt-5-mini          |                              |                              |
| gemini-embedding-001      | Google        | gpt-5.2             |                              |                              |
| gemini-embedding-001      | Google        | gpt-5               |                              |                              |
| gemini-embedding-001      | Google        | gpt-5-nano          |                              |                              |
| gemini-embedding-001      | Google        | gpt-5-mini          |                              |                              |
| text-embedding-3-large    | OpenAI        | gpt-5.2             |                              |                              |
| text-embedding-3-large    | OpenAI        | gpt-5               |                              |                              |
| text-embedding-3-large    | OpenAI        | gpt-5-nano          |                              |                              |
| text-embedding-3-large    | OpenAI        | gpt-5-mini          |                              |                              |
| Qwen3 0.6B                | Qwen          | gpt-5.2             |                              |                              |
| Qwen3 0.6B                | Qwen          | gpt-5               |                              |                              |
| Qwen3 0.6B                | Qwen          | gpt-5-nano          |                              |                              |
| Qwen3 0.6B                | Qwen          | gpt-5-mini          |                              |                              |
| text-embedding-3-small    | OpenAI        | gpt-5.2             |                              |                              |
| text-embedding-3-small    | OpenAI        | gpt-5               |                              |                              |
| text-embedding-3-small    | OpenAI        | gpt-5-nano          |                              |                              |
| text-embedding-3-small    | OpenAI        | gpt-5-mini          |                              |                              |
| Qwen3 4B                  | Qwen          | gpt-5.2             |                              |                              |
| Qwen3 4B                  | Qwen          | gpt-5               |                              |                              |
| Qwen3 4B                  | Qwen          | gpt-5-nano          |                              |                              |
| Qwen3 4B                  | Qwen          | gpt-5-mini          |                              |                              |