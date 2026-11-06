import io
import os
import shutil
import tempfile
import pytest
import pypdf
from src.rag import (
    extract_text_from_pdf,
    intelligent_chunk_document,
    AcademicVectorStore,
    answer_literature_query,
    answer_document_query,
    answer_document_chat,
    extract_key_information,
    compare_literature_methods,
    generate_research_summary,
)


@pytest.fixture
def sample_pdf_bytes():
    """Generate a clean 2-page academic PDF in memory for testing."""
    writer = pypdf.PdfWriter()

    # Page 1
    page1_text = (
        "Causal Inference in Observational Health Studies\n\n"
        "Abstract\n"
        "We investigate the Average Treatment Effect (ATE) of anti-hypertensive therapy using "
        "Augmented Inverse Probability Weighting (AIPW) and Propensity Score Matching. "
        "Under the assumption of conditional exchangeability and common support, the true treatment effect is identified."
    )
    writer.add_blank_page(width=612, height=792)

    # Page 2
    writer.add_blank_page(width=612, height=792)

    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


@pytest.fixture
def temp_chroma_dir():
    d = tempfile.mkdtemp(prefix="test_chroma_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_intelligent_chunking():
    pages = [
        {
            "page_num": 1,
            "title": "Rubin Causal Model 1974",
            "file_name": "rubin1974.pdf",
            "text": (
                "Section 1. Introduction to Potential Outcomes.\n\n"
                "Let Y(1) and Y(0) denote the potential outcomes under treatment and control. "
                "The fundamental problem of causal inference is that we only observe one potential outcome for each unit. "
                "Therefore, identification requires unconfoundedness and positivity."
            )
        },
        {
            "page_num": 2,
            "title": "Rubin Causal Model 1974",
            "file_name": "rubin1974.pdf",
            "text": (
                "Section 2. Identification Assumptions.\n\n"
                "We invoke SUTVA to ensure no interference between individual units. "
                "Under conditional exchangeability, adjusted outcome regression yields unbiased ATE estimates."
            )
        }
    ]

    chunks = intelligent_chunk_document(pages, target_chunk_size=150, overlap=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert "chunk_id" in c
        assert "page_num" in c
        assert c["page_num"] in [1, 2]
        assert "Rubin Causal Model" in c["title"]
        assert len(c["text"]) > 20


def test_academic_vector_store(temp_chroma_dir):
    store = AcademicVectorStore(persist_directory=temp_chroma_dir, collection_name="test_col")

    # Manually populate store
    pages = [
        {
            "page_num": 1,
            "title": "Double Machine Learning in Practice",
            "file_name": "chernozhukov2018.pdf",
            "text": "Double / Debiased Machine Learning eliminates first-order regularization bias using K-fold cross-fitting and Neyman orthogonality."
        },
        {
            "page_num": 3,
            "title": "Double Machine Learning in Practice",
            "file_name": "chernozhukov2018.pdf",
            "text": "The partially linear model Y = tau * T + g(X) + U estimates tau consistently when nuisance functions are estimated at rate n^{-1/4}."
        }
    ]
    chunks = intelligent_chunk_document(pages, target_chunk_size=300)

    # Insert into ChromaDB collection
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"title": c["title"], "page_num": c["page_num"], "file_name": c["file_name"], "char_count": len(c["text"])} for c in chunks]
    embeds = [store._get_embedding_vector(c["text"]) for c in chunks]

    store.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
    assert store.collection.count() >= 2

    # Query
    results = store.search_literature("Neyman orthogonality cross-fitting", top_k=2)
    assert len(results) >= 1
    assert "Double Machine Learning" in results[0]["title"]
    assert results[0]["page_num"] in [1, 3]

    # List papers
    paper_list = store.list_indexed_papers()
    assert len(paper_list) == 1
    assert paper_list[0]["title"] == "Double Machine Learning in Practice"


def test_rag_query_offline_fallback(temp_chroma_dir):
    store = AcademicVectorStore(persist_directory=temp_chroma_dir, collection_name="test_qa_col")

    pages = [{
        "page_num": 4,
        "title": "Rosenbaum Sensitivity Analysis",
        "file_name": "rosenbaum2002.pdf",
        "text": "Gamma bounds measure the sensitivity of matched pairs to unobserved confounding with odds ratio exceeding Gamma."
    }]
    chunks = intelligent_chunk_document(pages)
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"title": c["title"], "page_num": c["page_num"], "file_name": c["file_name"], "char_count": len(c["text"])} for c in chunks]
    embeds = [store._get_embedding_vector(c["text"]) for c in chunks]
    store.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    qa_res = answer_literature_query(
        query="What do Gamma bounds measure?",
        vector_store=store,
        api_key=None,  # Offline mode test
        top_k=1
    )
    assert "answer" in qa_res
    assert len(qa_res["retrieved_passages"]) == 1
    assert qa_res["retrieved_passages"][0]["page_num"] == 4
    assert "Rosenbaum Sensitivity Analysis" in qa_res["retrieved_passages"][0]["title"]


def test_literature_comparison_and_summary_offline(temp_chroma_dir):
    store = AcademicVectorStore(persist_directory=temp_chroma_dir, collection_name="test_summary_col")

    # Empty store case
    empty_comp = compare_literature_methods(store, "Causal Estimators")
    assert "upload at least one" in empty_comp["comparison_matrix"].lower()

    # With indexed document
    pages = [{
        "page_num": 1,
        "title": "E-Values for Observational Research",
        "file_name": "vanderweele2017.pdf",
        "text": "The E-value defines the minimum strength of association an unmeasured confounder must have with both treatment and outcome to explain away the effect."
    }]
    chunks = intelligent_chunk_document(pages)
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"title": c["title"], "page_num": c["page_num"], "file_name": c["file_name"], "char_count": len(c["text"])} for c in chunks]
    embeds = [store._get_embedding_vector(c["text"]) for c in chunks]
    store.collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeds
    )

    summary_out = generate_research_summary("E-Values for Observational Research", store, api_key=None)
    assert "summary" in summary_out
    assert "E-Values for Observational Research" in summary_out["summary"]


def test_multi_format_document_extraction():
    from src.rag import (
        extract_document_pages,
        extract_text_from_txt,
        extract_text_from_md,
        extract_text_from_csv,
        extract_text_from_docx,
        extract_text_from_excel,
        extract_text_from_pptx,
    )

    # 1. Plain Text
    txt_bytes = b"Research Note on Causal Mediation\n\nDirect effect and indirect effect decomposition via natural mediation formulas."
    txt_res = extract_text_from_txt(txt_bytes, "note.txt")
    assert len(txt_res) >= 1
    assert "Mediation" in txt_res[0]["text"]
    assert txt_res[0]["file_type"] == "txt"

    # 2. Markdown
    md_bytes = b"# Instrumental Variables Study\n\n## Identification\nZ satisfies relevance and exclusion restriction."
    md_res = extract_text_from_md(md_bytes, "iv_study.md")
    assert len(md_res) >= 1
    assert "Instrumental Variables Study" in md_res[0]["title"]
    assert md_res[0]["file_type"] == "md"

    # 3. CSV
    csv_bytes = b"treatment,outcome,age,income\n1,10.5,35,50000\n0,7.2,40,48000\n1,12.1,28,62000\n"
    csv_res = extract_text_from_csv(csv_bytes, "cohort.csv")
    assert len(csv_res) >= 2
    assert "Dataset Schema & Overview" in csv_res[0]["text"]
    assert csv_res[0]["file_type"] == "csv"

    # 4. DOCX (generate via python-docx)
    import docx
    doc = docx.Document()
    doc.add_heading("Doubly Robust Estimation in RWE", level=1)
    doc.add_paragraph("AIPW combines outcome modeling with propensity scores to achieve semiparametric efficiency.")
    docx_buf = io.BytesIO()
    doc.save(docx_buf)
    docx_bytes = docx_buf.getvalue()

    docx_res = extract_text_from_docx(docx_bytes, "aipw_rwe.docx")
    assert len(docx_res) >= 1
    assert "Doubly Robust Estimation in RWE" in docx_res[0]["title"]
    assert docx_res[0]["file_type"] == "docx"

    # 5. Excel (.xlsx generate via pandas / openpyxl)
    import pandas as pd
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        pd.DataFrame({"patient_id": [1, 2, 3], "treatment": [1, 0, 1], "hba1c_reduction": [1.4, 0.3, 1.2]}).to_excel(writer, sheet_name="Trial_Results", index=False)
    xlsx_bytes = excel_buf.getvalue()

    xlsx_res = extract_text_from_excel(xlsx_bytes, "trial_data.xlsx")
    assert len(xlsx_res) >= 1
    assert "Sheet 1: Trial_Results" in xlsx_res[0]["text"]
    assert xlsx_res[0]["file_type"] == "xlsx"

    # 6. PowerPoint (.pptx generate via python-pptx)
    import pptx
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_shape = slide.shapes.title
    title_shape.text = "Causal ML Symposium 2026"
    pptx_buf = io.BytesIO()
    prs.save(pptx_buf)
    pptx_bytes = pptx_buf.getvalue()

    pptx_res = extract_text_from_pptx(pptx_bytes, "presentation.pptx")
    assert len(pptx_res) >= 1
    assert "Causal ML Symposium" in pptx_res[0]["text"]
    assert pptx_res[0]["file_type"] == "pptx"


def test_multi_format_ingestion_and_deduplication(temp_chroma_dir):
    from src.rag import AcademicVectorStore, answer_document_query

    store = AcademicVectorStore(persist_directory=temp_chroma_dir, collection_name="test_multi_doc")

    class MockFile:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self.data = data
        def read(self):
            return self.data
        def seek(self, pos):
            pass

    file1 = MockFile("methods_guide.txt", b"Targeted Maximum Likelihood Estimation (TMLE) is a semiparametric substitution estimator.")
    file2 = MockFile("data_spec.md", b"# Clinical Cohort Protocol\n\nInclusion criteria require baseline eGFR > 60 mL/min/1.73m2.")

    # Ingest multiple files
    res = store.add_documents([file1, file2], api_key=None)
    assert res["documents_indexed"] == 2
    assert res["documents_skipped"] == 0
    assert store.collection.count() >= 2

    # Ingest again without force_reindex -> duplicate skip
    res2 = store.add_documents([file1, file2], api_key=None, force_reindex=False)
    assert res2["documents_indexed"] == 0
    assert res2["documents_skipped"] == 2

    # Query multi-format documents
    search_res = store.search_documents("Targeted Maximum Likelihood", top_k=2)
    assert len(search_res) >= 1
    assert "methods_guide.txt" in search_res[0]["file_name"]

    # Filter by format
    search_md = store.search_documents("Inclusion criteria", top_k=2, file_type_filter="md")
    assert len(search_md) >= 1
    assert search_md[0]["file_type"] == "md"

    # Query synthesis
    ans = answer_document_query("What are the inclusion criteria?", store, api_key=None, top_k=1)
    assert "answer" in ans
    assert len(ans["citations"]) >= 1

    # List documents
    docs = store.list_indexed_documents()
    assert len(docs) == 2
    types = [d["file_type"] for d in docs]
    assert "txt" in types
    assert "md" in types

    # Delete document
    del_ok = store.delete_document(docs[0]["doc_id"])
    assert del_ok is True
    assert len(store.list_indexed_documents()) == 1


def test_advanced_rag_workspace_features(temp_chroma_dir):
    store = AcademicVectorStore(persist_directory=temp_chroma_dir, collection_name="test_adv_rag")

    class MockFile:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self.data = data
        def seek(self, pos: int):
            pass
        def read(self) -> bytes:
            return self.data

    raw_text = (
        "Study Protocol: Evaluation of Metformin in Type 2 Diabetes.\n"
        "Sample size N = 2,450 adult patients across 12 academic medical centers.\n"
        "Primary exposure T is Metformin monotherapy vs standard lifestyle control.\n"
        "Primary outcome Y is 12-month change in HbA1c percentage points.\n"
        "Key confounders X include baseline age, BMI, systolic blood pressure, and eGFR.\n"
        "We employ Double Machine Learning with Gradient Boosted Trees and 5-fold cross-fitting.\n"
        "Estimated Average Treatment Effect is -0.74 percentage points (95% CI: -0.89 to -0.59, p < 0.001).\n"
        "Sensitivity analysis reveals an E-value of 2.14, demonstrating robustness to unmeasured confounding."
    )
    doc_file = MockFile("diabetes_trial_protocol.txt", raw_text.encode("utf-8"))

    # Ingest document
    res = store.add_documents([doc_file], target_chunk_size=200, overlap=30)
    assert res["documents_indexed"] == 1
    doc_id = res["skipped_files"][0]["doc_id"] if res["skipped_files"] else store.list_indexed_documents()[0]["doc_id"]

    # 1. Test Document Preview
    preview = store.get_document_preview(doc_id)
    assert preview is not None
    assert "diabetes_trial_protocol.txt" in preview["file_name"]
    assert preview["total_chunks"] >= 1
    assert len(preview["chunks_preview"]) >= 1
    assert "diabetes" in preview["chunks_preview"][0]["text"].lower()

    # 2. Test Document Re-indexing
    reindex_res = store.reindex_document(doc_id, target_chunk_size=400, overlap=50)
    assert reindex_res["success"] is True
    assert reindex_res["new_chunks_count"] >= 1

    # Reindex all
    reindex_all_res = store.reindex_all_documents(target_chunk_size=350, overlap=40)
    assert reindex_all_res["total_documents_reindexed"] == 1

    # 3. Test Conversational Document Chat (Follow-up Support)
    history = [
        {"role": "user", "content": "What is the primary intervention evaluated in this study?"},
        {"role": "assistant", "content": "The primary intervention evaluated is Metformin monotherapy."}
    ]
    chat_res = answer_document_chat(
        query="What was the estimated effect on HbA1c and sample size?",
        vector_store=store,
        chat_history=history,
        api_key=None,  # Offline test
        top_k=2
    )
    assert "answer" in chat_res
    assert len(chat_res["citations"]) >= 1
    assert "HbA1c" in chat_res["answer"] or "diabetes" in chat_res["answer"].lower()

    # 4. Test Insufficient Evidence Response when query is unrelated and threshold is strict
    insufficient_res = answer_document_chat(
        query="Quantum gravity string theory supersymmetry cosmological constant",
        vector_store=store,
        api_key=None,
        min_similarity=0.99  # Intentionally strict threshold
    )
    assert insufficient_res["insufficient_evidence"] is True
    assert "not contain sufficient evidence" in insufficient_res["answer"].lower()

    # 5. Test Key Information Extraction
    info_res = extract_key_information(doc_id, store, api_key=None)
    assert "markdown" in info_res
    assert "diabetes" in info_res["markdown"].lower() or "Target Document" in info_res["markdown"]


def test_complete_rag_pipeline_with_pdf_docx_csv(tmp_path, monkeypatch):
    """
    Test the complete RAG pipeline with sample PDF, DOCX, and CSV files,
    verifying multi-format extraction, ChromaDB persistence, and genuine evidence retrieval.
    """
    persist_dir = str(tmp_path / "custom_chroma")
    monkeypatch.setenv("CHROMADB_PERSIST_DIR", persist_dir)

    # Initialize store using environment-configured persist_directory
    store = AcademicVectorStore(collection_name="pipeline_test_col")
    assert store.persist_directory == persist_dir

    # 1. Sample PDF Document
    raw_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 120 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Pearl SCM Backdoor Criterion: Directed acyclic graphs provide a sound graphical criterion to identify non-causal paths.) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000247 00000 n \n0000000418 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n491\n%%EOF\n"
    )

    # 2. Sample DOCX Document
    import docx
    doc = docx.Document()
    doc.add_heading("Double Machine Learning for Causal Inference", level=1)
    doc.add_paragraph("Neyman orthogonality and cross-fitting remove first-order bias from machine learning regularizers.")
    docx_buf = io.BytesIO()
    doc.save(docx_buf)
    raw_docx = docx_buf.getvalue()

    # 3. Sample CSV Document
    raw_csv = (
        b"patient_id,treatment,sbp_reduction,age,baseline_risk\n"
        b"101,1,14.5,58,High\n"
        b"102,0,2.1,62,High\n"
        b"103,1,16.2,49,Medium\n"
        b"104,0,3.0,55,Low\n"
    )

    # Ingest all three formats
    res_pdf = store.ingest_document(raw_pdf, "pearl_backdoor.pdf")
    res_docx = store.ingest_document(raw_docx, "dml_causal.docx")
    res_csv = store.ingest_document(raw_csv, "hypertension_trial.csv")

    assert res_pdf["success"] is True
    assert res_docx["success"] is True
    assert res_csv["success"] is True

    indexed_docs = store.list_indexed_documents()
    assert len(indexed_docs) == 3
    doc_types = {d["file_type"] for d in indexed_docs}
    assert "pdf" in doc_types
    assert "docx" in doc_types
    assert "csv" in doc_types

    # Query 1: Retrieve from PDF
    pdf_qa = answer_document_chat("What does the backdoor criterion provide?", store, api_key=None, top_k=2)
    assert "answer" in pdf_qa
    assert len(pdf_qa["citations"]) >= 1
    assert any("pearl_backdoor.pdf" in c["file_name"] for c in pdf_qa["citations"])
    assert "backdoor" in pdf_qa["answer"].lower()

    # Query 2: Retrieve from DOCX
    docx_qa = answer_document_chat("How does Double Machine Learning remove regularization bias?", store, api_key=None, top_k=2)
    assert "answer" in docx_qa
    assert len(docx_qa["citations"]) >= 1
    assert any("dml_causal.docx" in c["file_name"] for c in docx_qa["citations"])
    assert "orthogonality" in docx_qa["answer"].lower() or "cross-fitting" in docx_qa["answer"].lower()

    # Query 3: Retrieve from CSV
    csv_qa = answer_document_chat("What clinical variables and columns are in the hypertension trial dataset?", store, api_key=None, top_k=2)
    assert "answer" in csv_qa
    assert len(csv_qa["citations"]) >= 1
    assert any("hypertension_trial.csv" in c["file_name"] for c in csv_qa["citations"])
    assert "sbp_reduction" in csv_qa["answer"].lower() or "hypertension" in csv_qa["answer"].lower()



