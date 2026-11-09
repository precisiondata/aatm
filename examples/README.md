# Examples

Short, runnable notebooks, in reading order. Each one has its real output saved, so you
can read it without running anything.

| Notebook | You learn | Table | Mapper |
|---|---|---|---|
| [01_terminology_mapping](01_terminology_mapping.ipynb) | Map free text to standard concepts; default vs. LLM selector; domain and vocabulary filters | – | yes |
| [02_omop_quickstart](02_omop_quickstart.ipynb) | Notes → OMOP rows with a predefined definition (ECOG); provenance; issues | `measurement` | no |
| [03_custom_entity_definition](03_custom_entity_definition.ipynb) | Write your own YAML definition; validate codes before model calls; UCUM unit lookup | `measurement` | no |
| [04_mapper_configuration](04_mapper_configuration.ipynb) | Code extracted text with the mapper; review mappings; `vocabularies` and `min_score`; rebuild without re-extracting | `measurement` | yes |
| [05_observation](05_observation.ipynb) | Combine several definitions in one run (smoking, family and medical history) | `observation` | yes |
| [06_condition_occurrence](06_condition_occurrence.ipynb) | Diagnoses with a status; what the CDM leaves out | `condition_occurrence` | yes |
| [07_drug_exposure](07_drug_exposure.ipynb) | Drugs by kind (administered, prescribed, taking); inferred end dates; routes | `drug_exposure` | yes |
| [08_device_exposure](08_device_exposure.ipynb) | Devices by kind (placed, existing); quantity | `device_exposure` | yes |

## Before you run them

1. Install the project with its dev dependencies (`uv sync`), which include Jupyter's
   `ipykernel`.
2. Download the OMOP vocabularies from [Athena](https://athena.ohdsi.org/vocabulary/list)
   (at least SNOMED, LOINC, UCUM and RxNorm) and run `aatm init` from the project root.
   The notebooks use the `text-embedding-3-small` index: if you chose another embedding
   model, change the `retriever=` argument.
3. Put `OPENAI_API_KEY` in a `.env` file at the project root. The notebooks call
   `gpt-5-mini`; any provider-prefixed model id works, e.g. `gemini/gemini-2.5-flash`.

The first cell of each notebook moves to the project root, where `.aatm/` and `.env` live,
so you can open the notebooks from `examples/` or from the root. LLM outputs vary a little
between runs.

More detail: [From notes to OMOP tables](https://precisiondata.github.io/aatm/getting-started/omop_tables/).
