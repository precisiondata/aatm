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