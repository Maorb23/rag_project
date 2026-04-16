from __future__ import annotations

from typing import Any

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

    def _retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        docs = self.vectorstore.similarity_search(query, k=k)
        retrieved: list[dict[str, Any]] = []
        for doc in docs:
            retrieved.append(
                {
                    "doc_name": doc.metadata.get("doc_name"),
                    "page_number": doc.metadata.get("page_number"),
                    "content": doc.page_content,
                }
            )
        return retrieved

    @staticmethod
    def _format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
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

    def answer_with_rag(self, query: str, k: int = 4) -> dict[str, Any]:
        retrieved_chunks = self._retrieve(query=query, k=k)
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
    ) -> pd.DataFrame:
        use_k = k or self.config.retrieval_default_k
        rows: list[dict[str, Any]] = []

        for item in questions_df.to_dict(orient="records"):
            result = self.answer_with_rag(str(item.get("question", "")), k=use_k)
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
