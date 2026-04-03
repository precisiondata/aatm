from enum import Enum
import dotenv
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi

# Custom modules
from aatm.data_models import RetrieverResults, Translation
from aatm.pipeline import PipelineBaseClass

# Load environment variables
dotenv.load_dotenv()


class BaseReranker(PipelineBaseClass, ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def rerank(self, retriever_results: RetrieverResults) -> Translation:
        pass

    def __call__(self, retriever_results: RetrieverResults) -> RetrieverResults:
        assert isinstance(retriever_results, RetrieverResults), (
            f"retriever_results must be of type RetrieverResults. Got {type(retriever_results)}."
        )

        return self.rerank(retriever_results)


class BM25Reranker(BaseReranker):
    def rerank(self, retriever_results: RetrieverResults) -> RetrieverResults:
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
    QWEN3_06B = "Qwen/Qwen3-Reranker-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Reranker-4B"
    QWEN3_8B = "Qwen/Qwen3-Reranker-8B"


class Qwen3Reranker(BaseReranker):
    def __init__(
        self,
        model_id: str,
        max_length: int = 8192,
        task: str = None,
        prefix: str = None,
        suffix: str = None,
        *args,
        **kwargs,
    ):
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

    def format_instruction(self, instruction, query, doc):
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
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
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, retriever_results):
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


RERANKER_REGISTRY = {
    "BM25": BM25Reranker,
    Qwen3RerankerModels.QWEN3_06B.value: Qwen3Reranker,
    Qwen3RerankerModels.QWEN3_4B.value: Qwen3Reranker,
    Qwen3RerankerModels.QWEN3_8B.value: Qwen3Reranker,
}


def load_reranker(reranker_name, **kwargs):
    return RERANKER_REGISTRY[reranker_name](reranker_name, **kwargs)
