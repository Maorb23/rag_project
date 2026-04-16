from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import PipelineConfig

RAW_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


def _normalize_pdf_name(doc_name: str) -> str:
    name = str(doc_name).strip()
    if name.lower().endswith(".pdf"):
        return name
    return f"{name}.pdf"


def download_required_pdfs(doc_names: Iterable[str], pdf_dir: Path, timeout: int = 60) -> list[Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    downloaded = 0
    existing = 0
    missing = 0

    unique_doc_names = sorted({str(x).strip() for x in doc_names if str(x).strip()})
    total = len(unique_doc_names)
    print(f"Preparing PDFs: {total} files referenced")

    for i, doc_name in enumerate(unique_doc_names, start=1):
        pdf_name = _normalize_pdf_name(doc_name)
        local_path = pdf_dir / pdf_name
        if not local_path.exists():
            url = f"{RAW_PDF_BASE}/{pdf_name}"
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200:
                missing += 1
                if i <= 20 or i % 50 == 0:
                    print(f"[{i}/{total}] Missing on source: {pdf_name}")
                continue
            local_path.write_bytes(response.content)
            downloaded += 1
            if i <= 20 or i % 50 == 0:
                print(f"[{i}/{total}] Downloaded: {pdf_name}")
        else:
            existing += 1
            if i <= 20 or i % 50 == 0:
                print(f"[{i}/{total}] Already local: {pdf_name}")
        local_paths.append(local_path)

    print(
        "PDF summary - "
        f"downloaded: {downloaded}, existing: {existing}, missing: {missing}, usable local: {len(local_paths)}"
    )
    return local_paths


def load_pdf_pages_with_metadata(doc_table: pd.DataFrame, pdf_dir: Path) -> list[Document]:
    required = {"doc_name", "company", "doc_period"}
    missing = required - set(doc_table.columns)
    if missing:
        raise ValueError(f"Missing columns in doc_table: {sorted(missing)}")

    all_pages: list[Document] = []
    skipped: list[str] = []
    rows = doc_table.to_dict(orient="records")
    total = len(rows)
    print(f"Loading pages from local PDFs: {total} documents")

    for i, row in enumerate(rows, start=1):
        pdf_name = _normalize_pdf_name(row["doc_name"])
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            if i <= 20 or i % 50 == 0:
                print(f"[{i}/{total}] Skipped missing local file: {pdf_name}")
            continue

        loader = PyPDFLoader(str(pdf_path))
        try:
            pages = loader.load()
            if i <= 20 or i % 50 == 0:
                print(f"[{i}/{total}] Loaded {len(pages)} pages from {pdf_name}")
        except Exception as exc:
            error_text = f"{pdf_name}: {exc}"
            skipped.append(error_text)
            print(f"[{i}/{total}] Failed to load {error_text}")
            continue

        for page in pages:
            page_number = int(page.metadata.get("page", 0))
            page.metadata["doc_name"] = row["doc_name"]
            page.metadata["company"] = row["company"]
            page.metadata["doc_period"] = row["doc_period"]
            page.metadata["page_number"] = page_number
            page.metadata.pop("page", None)

        all_pages.extend(pages)

    if skipped:
        print("Skipped PDFs during load due to read/encryption issues:")
        for item in skipped[:20]:
            print(f"- {item}")
        if len(skipped) > 20:
            print(f"... and {len(skipped) - 20} more")

    print(f"Total loaded pages: {len(all_pages)}")

    return all_pages


def chunk_documents(
    pages: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def build_or_load_vectorstore(
    chunks: list[Document],
    config: PipelineConfig,
    force_rebuild: bool = False,
) -> FAISS:
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    index_file = config.vectorstore_dir / "index.faiss"

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 2, "normalize_embeddings": True},
    )

    if index_file.exists() and not force_rebuild:
        print(f"Loading existing vectorstore from {config.vectorstore_dir} ...")
        return FAISS.load_local(
            str(config.vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not chunks:
        raise ValueError("No chunks available to index. Cannot build vectorstore.")

    print(f"Building new vectorstore from {len(chunks)} chunks ...")
    batch_size = 64

    first_batch = chunks[:batch_size]
    vectorstore = FAISS.from_documents(first_batch, embeddings)
    print(f"Indexed initial batch: {len(first_batch)}/{len(chunks)}")

    for start in range(batch_size, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vectorstore.add_documents(chunks[start:end])
        print(f"Indexed batch: {end}/{len(chunks)}")

    vectorstore.save_local(str(config.vectorstore_dir))
    print(f"Saved vectorstore to {config.vectorstore_dir}")
    return vectorstore
