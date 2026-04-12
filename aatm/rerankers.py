"""
Define reranker abstractions and implementations for retrieval result refinement.

This module provides base classes and concrete reranker implementations for
post-processing retrieval results in a terminology-mapping pipeline. It includes
a simple BM25-based lexical reranker and a neural reranker based on Qwen3
causal language models.

The module is designed around pipeline-style composition, allowing rerankers to
be used as callable processing stages that accept retrieval outputs and return
the same results reordered by reranking scores.
"""

from enum import Enum
from typing import Any, List
import dotenv
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi

# Custom modules
from aatm.data_models import RetrieverResults
from aatm.pipeline import PipelineBaseClass

# Load environment variables
dotenv.load_dotenv()


class BaseReranker(PipelineBaseClass, ABC):
    """Define the abstract interface for reranker pipeline components.

    This base class establishes the contract for reranker implementations that
    operate on retrieval results. Subclasses must implement the ``rerank()``
    method, while the base ``__call__()`` method provides runtime type checking
    and pipeline-compatible invocation behavior.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the reranker base class.

        This constructor accepts arbitrary positional and keyword arguments to
        support a flexible subclass interface, but it does not perform any
        initialization itself.

        Args:
            *args: Positional arguments reserved for subclasses.
            **kwargs: Keyword arguments reserved for subclasses.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def rerank(self, retriever_results: RetrieverResults) -> RetrieverResults:
        """Rerank the retrieved results for one or more queries.

        Subclasses must implement this method to assign reranking scores and
        reorder documents within each query result set according to their
        relevance.

        Args:
            retriever_results: Retrieval output containing the original queries
                and their associated candidate results.

        Returns:
            The reranked retrieval results.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        pass

    def __call__(self, retriever_results: RetrieverResults) -> RetrieverResults:
        """Validate the input type and rerank the retrieval results.

        This method makes reranker instances directly callable and compatible
        with the pipeline interface. It checks that the input is a
        ``RetrieverResults`` instance before delegating to ``rerank()``.

        Args:
            retriever_results: Retrieval output to be reranked.

        Returns:
            The reranked retrieval results.

        Raises:
            AssertionError: If ``retriever_results`` is not an instance of
                ``RetrieverResults``.
        """
        assert isinstance(retriever_results, RetrieverResults), (
            f"retriever_results must be of type RetrieverResults. Got {type(retriever_results)}."
        )

        return self.rerank(retriever_results)


class BM25Reranker(BaseReranker):
    """Rerank retrieved documents using BM25 lexical similarity.

    This reranker scores each retrieved document against its corresponding query
    using BM25 over whitespace-tokenized text. Scores are stored in each result
    object and the result lists are sorted in descending score order.
    """

    def rerank(self, retriever_results: RetrieverResults) -> RetrieverResults:
        """Rerank retrieval results using BM25 scores.

        For each query, this method builds a BM25 index over the retrieved
        document expressions, computes lexical relevance scores for the query,
        stores the scores in the corresponding result objects, and sorts each
        result list by descending rerank score.

        Args:
            retriever_results: Retrieval output containing queries and their
                associated candidate documents.

        Returns:
            The same ``RetrieverResults`` object with updated ``rerank_score``
                values and results reordered by BM25 relevance.
        """
        for query_index, query in enumerate(retriever_results.queries):
            corpus = retriever_results.results[query_index]
            tokenized_corpus = [doc.expression.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)

            # update scores
            for doc_index, _ in enumerate(corpus):
                retriever_results.results[query_index][
                    doc_index
                ].rerank_score = doc_scores[doc_index]

        # sort results based on scores
        for list_of_results in retriever_results.results:
            list_of_results.sort(key=lambda x: x.rerank_score, reverse=True)

        return retriever_results


class Qwen3RerankerModels(Enum):
    """Enumerate the supported Qwen3 reranker model identifiers.

    This enumeration provides the Hugging Face model names corresponding to the
    available Qwen3 reranker variants supported by the package.
    """

    QWEN3_06B = "Qwen/Qwen3-Reranker-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Reranker-4B"
    QWEN3_8B = "Qwen/Qwen3-Reranker-8B"


class Qwen3Reranker(BaseReranker):
    """Rerank retrieved documents with a Qwen3 language-model-based judge.

    This reranker formats each query-document pair as an instruction-following
    relevance judgment task, runs the pairs through a Qwen3 causal language
    model, and interprets the model's probability of answering "yes" as the
    reranking score.

    The class supports custom task instructions, prompt prefix and suffix
    templates, configurable sequence length, and automatic placement on CPU or
    CUDA when available.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 8192,
        task: str = None,
        prefix: str = None,
        suffix: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen3 reranker and load the model resources.

        This constructor resolves the requested model identifier, loads the
        tokenizer and causal language model, configures the execution device,
        defines the task prompt and prompt wrappers, and precomputes tokenized
        prefix and suffix sequences used during input construction.

        Args:
            model_id: Name of the supported Qwen3 reranker model variant.
            max_length: Maximum total token length for each formatted input,
                including prefix and suffix tokens.
            task: Optional task instruction describing the relevance judgment
                objective. A default instruction is used when not provided.
            prefix: Optional prompt prefix inserted before each formatted query-
                document pair. A default system-and-user prompt is used when not
                provided.
            suffix: Optional prompt suffix appended after each formatted query-
                document pair. A default assistant prompt is used when not
                provided.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.

        Raises:
            ValueError: If ``model_id`` is not a valid member of
                ``Qwen3RerankerModels``.
            OSError: If the tokenizer or model weights cannot be loaded.
        """
        self.model_id = Qwen3RerankerModels(model_id).value
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.task = task
        self.prefix = prefix
        self.suffix = suffix
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        if self.task is None:
            self.task = (
                "Given a search query, retrieve relevant concepts that match the query."
            )

        if self.prefix is None:
            self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'

        if self.suffix is None:
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format a query-document pair as a reranking instruction string.

        This helper builds the textual input passed to the language model by
        combining the task instruction, query, and candidate document into a
        structured prompt.

        Args:
            instruction: Task-level instruction that defines the relevance
                judgment objective.
            query: Search query or source expression being matched.
            doc: Candidate document or concept expression to evaluate.

        Returns:
            A formatted string containing the instruction, query, and document.
        """
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs: List[str]) -> dict[str, torch.Tensor]:
        """Tokenize and pad formatted query-document pairs for model inference.

        This method tokenizes the provided text pairs, applies truncation while
        reserving space for the configured prefix and suffix tokens, appends
        those tokens to each sequence, pads the batch, and moves the resulting
        tensors to the model device.

        Args:
            pairs: Sequence of formatted query-document prompt strings.

        Returns:
            A dictionary of model input tensors suitable for forward inference.
        """
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(
        self, inputs: dict[str, torch.Tensor], **kwargs: Any
    ) -> list[float]:
        """Compute relevance scores from the model's final-token logits.

        This method runs the model in inference mode, extracts the logits for
        the final token position, compares the logits for the "yes" and "no"
        tokens, applies a log-softmax over those two values, and returns the
        probability assigned to "yes" for each input example.

        Args:
            inputs: Tokenized model inputs prepared for batch inference.
            **kwargs: Additional keyword arguments reserved for future
                extensions.

        Returns:
            A list of relevance scores, where each score is the model's
            probability that the corresponding document is relevant to the
            query.
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, retriever_results: RetrieverResults) -> RetrieverResults:
        """Rerank retrieval results using Qwen3 relevance judgments.

        This method converts each query-document pair into an instruction-based
        prompt, performs batched model inference to obtain relevance scores,
        assigns those scores to the corresponding retrieved documents, and sorts
        each query result list in descending score order.

        Args:
            retriever_results: Retrieval output containing queries and candidate
                documents to rerank.

        Returns:
            The same retrieval results object with updated ``rerank_score``
            values and reordered candidate lists.
        """
        # create pairs with queries and retrieved docs
        pairs = []
        n_results_per_query = len(retriever_results.results[0])
        for query, retrieved_docs in zip(
            retriever_results.queries, retriever_results.results
        ):
            for doc in retrieved_docs:
                pairs.append(self.format_instruction(self.task, query, doc.expression))

        # compute scores
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        # assign scores
        query_index = 0
        result_index = 0
        for i, score in enumerate(scores):
            if i % n_results_per_query == 0 and i != 0:
                query_index += 1
                result_index = 0
            retriever_results.results[query_index][result_index].rerank_score = score
            result_index += 1

        # sort results based on scores
        for list_of_results in retriever_results.results:
            list_of_results.sort(key=lambda x: x.rerank_score, reverse=True)

        return retriever_results
