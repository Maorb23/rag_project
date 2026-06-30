from __future__ import annotations

from typing import Any
from operator import itemgetter

from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
from langchain_community.vectorstores import FAISS

from .config import PipelineConfig
from .nebius_client import NebiusChatClient


SYSTEM_PROMPT = (
    "You are a financial QA assistant. Use only the provided context. "
    "If context does not contain the answer, say explicitly that the context is insufficient. "
    "Keep answers concise and cite source document names for factual claims."
)

class RAGPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        vectorstore: FAISS,
        client: NebiusChatClient | None = None,
    ) -> None:
        self.config = config
        self.vectorstore = vectorstore
        self.client = client or NebiusChatClient(config)
        self._rerank_embeddings: HuggingFaceEmbeddings | None = None

    def _get_rerank_embeddings(self) -> HuggingFaceEmbeddings:
        if self._rerank_embeddings is None:
            self._rerank_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 8, "normalize_embeddings": True},
            )
        return self._rerank_embeddings

    def _retrieve(self, query: str, k: int, rerank: bool | None = None) -> list[dict[str, Any]]:
        """
        Function to retrieve relevant document chunks from the vectorstore based on the input query.
        Returns a list of dictionaries containing the retrieved chunk's content and metadata (doc_name, page_number).
        """
        # determine whether to rerank: explicit arg overrides config
        use_rerank = bool(self.config.rerank_enabled) if rerank is None else bool(rerank)

        # If using rerank, first retrieve a larger candidate set
        candidate_k = k
        if use_rerank:
            candidate_k = max(k, int(self.config.rerank_top_k or 20))

        docs = self.vectorstore.similarity_search(query, k=candidate_k)

        retrieved: list[dict[str, Any]] = []
        for doc in docs:
            retrieved.append(
                {
                    "doc_name": doc.metadata.get("doc_name"),
                    "page_number": doc.metadata.get("page_number"),
                    "content": doc.page_content,
                }
            )

        # If reranking is enabled, rescore candidates with the embedding model and return top final_k
        if use_rerank and retrieved:
            try:
                emb = self._get_rerank_embeddings()

                query_vec = emb.embed_query(query)
                docs_texts = [r["content"] for r in retrieved]
                docs_vecs = emb.embed_documents(docs_texts)

                # compute dot-product scores (embeddings are normalized if model configured so)
                scores = []
                for vec in docs_vecs:
                    # dot product
                    score = sum(q * d for q, d in zip(query_vec, vec))
                    scores.append(score)

                for r, s in zip(retrieved, scores):
                    r["rerank_score"] = float(s)

                final_k = int(self.config.rerank_final_k or k)
                # pick top final_k by rerank_score
                retrieved = sorted(retrieved, key=itemgetter("rerank_score"), reverse=True)[:final_k]
            except Exception:
                # on any rerank failure, fall back to original ordering
                pass

        # if not reranking, keep top-k from initial retrieval
        if not use_rerank:
            retrieved = retrieved[:k]

        return retrieved

    @staticmethod
    def _format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
        """
        Function to format the retrieved document chunks into a single string context for the language model.
        Each chunk is separated by a clear delimiter, and includes its source document name and page number for reference.
        If no chunks are retrieved, returns a message indicating that no context is available.
        """
        if not retrieved_chunks:
            return "No retrieved context is available for this question."

        blocks: list[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Chunk {i}",
                        f"doc_name: {chunk.get('doc_name')}",
                        f"page_number: {chunk.get('page_number')}",
                        "---",
                        str(chunk.get("content", "")),
                    ]
                )
            )
        return "\n\n-----\n\n".join(blocks)

    def answer_with_rag(self, query: str, k: int = 4, rerank: bool | None = None) -> dict[str, Any]:
        retrieved_chunks = self._retrieve(query=query, k=k, rerank=rerank)
        context = self._format_context(retrieved_chunks)

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer based only on the context. If not answerable from context, say so clearly."
        )

        answer = self.client.chat(
            model=self.config.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        minimal_chunks = [
            {
                "doc_name": c.get("doc_name"),
                "page_number": c.get("page_number"),
            }
            for c in retrieved_chunks
        ]

        return {
            "answer": answer,
            "retrieved_chunks": minimal_chunks,
        }

    def run_on_dataframe(
        self,
        questions_df: pd.DataFrame,
        k: int | None = None,
        rerank: bool | None = None,
    ) -> pd.DataFrame:
        use_k = k or self.config.retrieval_default_k
        use_rerank = self.config.rerank_enabled if rerank is None else bool(rerank)
        rows: list[dict[str, Any]] = []

        for item in questions_df.to_dict(orient="records"):
            result = self.answer_with_rag(str(item.get("question", "")), k=use_k, rerank=use_rerank)
            rows.append(
                {
                    "financebench_id": item.get("financebench_id"),
                    "question": item.get("question"),
                    "ground_truth": item.get("answer", ""),
                    "rag_answer": result["answer"],
                    "retrieved_chunks": result["retrieved_chunks"],
                }
            )

        return pd.DataFrame(rows)
