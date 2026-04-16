from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .comparison import build_side_by_side_table
from .config import PipelineConfig
from .dataset import build_doc_metadata_table, prepare_stage1_dataset
from .evaluation import (
    compute_correctness_judgements,
    compute_faithfulness_first_20,
    compute_page_hit_rate,
)
from .io_utils import save_dataframe_csv, save_dataframe_json, save_json
from .naive_generation import run_naive_generation, sample_stage2_questions
from .rag_pipeline import RAGPipeline
from .retrieval_checks import run_retrieval_sanity_checks
from .vectorstore import (
    build_or_load_vectorstore,
    chunk_documents,
    download_required_pdfs,
    load_pdf_pages_with_metadata,
)


def execute_full_pipeline(config: PipelineConfig) -> dict[str, Any]:
    config.results_dir.mkdir(parents=True, exist_ok=True)

    filtered_df, doc_mapping_df = prepare_stage1_dataset(
        dataset_id=config.dataset_id,
        split=config.dataset_split,
        output_dir=config.results_dir,
    )

    save_dataframe_csv(filtered_df, config.results_dir / "stage1_filtered.csv")
    save_dataframe_json(filtered_df, config.results_dir / "stage1_filtered.json")
    save_dataframe_csv(doc_mapping_df, config.results_dir / "stage1_doc_mapping.csv")
    save_dataframe_json(doc_mapping_df, config.results_dir / "stage1_doc_mapping.json")

    stage2_questions = sample_stage2_questions(filtered_df, n_per_type=5)
    naive_df = run_naive_generation(stage2_questions, config)
    save_dataframe_csv(naive_df, config.results_dir / "stage2_naive.csv")
    save_dataframe_json(naive_df, config.results_dir / "stage2_naive.json")

    doc_table = build_doc_metadata_table(filtered_df)
    download_required_pdfs(doc_table["doc_name"].tolist(), config.pdf_dir)
    pages = load_pdf_pages_with_metadata(doc_table, config.pdf_dir)
    chunks = chunk_documents(
        pages,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    vectorstore = build_or_load_vectorstore(chunks, config)

    sanity_input = filtered_df.head(3)
    sanity_df = run_retrieval_sanity_checks(sanity_input, vectorstore, k=config.retrieval_default_k)
    save_dataframe_csv(sanity_df, config.results_dir / "stage3_sanity_checks.csv")
    save_dataframe_json(sanity_df, config.results_dir / "stage3_sanity_checks.json")

    rag = RAGPipeline(config=config, vectorstore=vectorstore)

    rag_stage2_df = rag.run_on_dataframe(stage2_questions, k=config.retrieval_default_k)
    save_dataframe_csv(rag_stage2_df, config.results_dir / "stage4_rag_stage2_questions.csv")
    save_dataframe_json(rag_stage2_df, config.results_dir / "stage4_rag_stage2_questions.json")

    comparison_df = build_side_by_side_table(stage2_questions, naive_df, rag_stage2_df)
    save_dataframe_csv(comparison_df, config.results_dir / "stage5_comparison.csv")
    save_dataframe_json(comparison_df, config.results_dir / "stage5_comparison.json")

    rag_full_df = rag.run_on_dataframe(filtered_df, k=config.retrieval_default_k)
    save_dataframe_csv(rag_full_df, config.results_dir / "stage6_rag_full.csv")
    save_dataframe_json(rag_full_df, config.results_dir / "stage6_rag_full.json")

    correctness_df, correctness_score = compute_correctness_judgements(rag_full_df, config)
    save_dataframe_csv(correctness_df, config.results_dir / "stage6_correctness.csv")
    save_dataframe_json(correctness_df, config.results_dir / "stage6_correctness.json")

    faithfulness_df, faithfulness_score = compute_faithfulness_first_20(rag_full_df, config)
    save_dataframe_csv(faithfulness_df, config.results_dir / "stage6_faithfulness.csv")
    save_dataframe_json(faithfulness_df, config.results_dir / "stage6_faithfulness.json")

    hit_detail_df, hit_summary_df = compute_page_hit_rate(
        questions_df=filtered_df,
        rag_pipeline=rag,
        k_values=config.retrieval_hit_k_values,
    )
    save_dataframe_csv(hit_detail_df, config.results_dir / "stage6_hit_detail.csv")
    save_dataframe_json(hit_detail_df, config.results_dir / "stage6_hit_detail.json")
    save_dataframe_csv(hit_summary_df, config.results_dir / "stage6_hit_summary.csv")
    save_dataframe_json(hit_summary_df, config.results_dir / "stage6_hit_summary.json")

    metrics = {
        "correctness_accuracy": correctness_score,
        "faithfulness_mean": faithfulness_score,
        "page_hit_rates": hit_summary_df.to_dict(orient="records"),
    }
    save_json(metrics, config.results_dir / "stage6_metrics_summary.json")

    return {
        "filtered_df": filtered_df,
        "stage2_questions": stage2_questions,
        "naive_df": naive_df,
        "rag_stage2_df": rag_stage2_df,
        "comparison_df": comparison_df,
        "correctness_df": correctness_df,
        "faithfulness_df": faithfulness_df,
        "hit_detail_df": hit_detail_df,
        "hit_summary_df": hit_summary_df,
        "metrics": metrics,
    }
