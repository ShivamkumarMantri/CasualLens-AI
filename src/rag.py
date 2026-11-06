import hashlib
import io
import mimetypes
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import pypdf
from src.gemini_client import (
    get_api_key,
    generate_gemini_text,
    generate_gemini_embedding,
    mask_api_key,
    sanitize_message,
)

_ENV_DEFAULT = object()

# Optional document parsing libraries with graceful fallbacks
try:
    import docx
    HAS_DOCX = True
except ImportError:
    docx = None
    HAS_DOCX = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    openpyxl = None
    HAS_OPENPYXL = False

try:
    import pptx
    HAS_PPTX = True
except ImportError:
    pptx = None
    HAS_PPTX = False


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".txt": "txt",
    ".md": "md",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
}


def compute_file_hash(data: bytes) -> str:
    """Compute deterministic SHA-256 fingerprint (16-char hex prefix) of file bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def read_input_bytes(file_input: Any, default_name: str = "document") -> Tuple[bytes, str]:
    """Helper to extract raw bytes and filename from file path, bytes, or Streamlit UploadedFile."""
    file_name = default_name
    if isinstance(file_input, (str, os.PathLike)):
        file_name = os.path.basename(file_input)
        with open(file_input, "rb") as f:
            return f.read(), file_name
    elif isinstance(file_input, bytes):
        return file_input, file_name
    elif hasattr(file_input, "read"):
        file_name = getattr(file_input, "name", default_name)
        file_input.seek(0)
        data = file_input.read()
        return data, file_name
    return b"", file_name


def clean_text(text: str) -> str:
    """Clean and normalize extracted document text."""
    if not text:
        return ""
    # Fix hyphenation across linebreaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Remove null bytes and non-printable control characters (except \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize tabs to spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize consecutive linebreaks
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ==============================================================================
# SECTION CLASSIFICATION
# ==============================================================================
SECTION_KEYWORDS = [
    (r"(?i)\babstract\b", "Abstract"),
    (r"(?i)\b(introduction|background|motivation|overview)\b", "Introduction"),
    (r"(?i)\b(identification|structural causal|assumptions|unconfoundedness|positivity|sutva|backdoor)\b", "Identification & Assumptions"),
    (r"(?i)\b(methodology|methods|estimators|dml|aipw|psm|ipw|regression|architecture|model)\b", "Methodology & Estimators"),
    (r"(?i)\b(empirical results|results|findings|simulation|monte carlo|metrics|evaluation)\b", "Empirical Results"),
    (r"(?i)\b(sensitivity|e-value|robustness|bounds|refutation|validation)\b", "Sensitivity & Robustness"),
    (r"(?i)\b(discussion|conclusion|limitations|future work|summary)\b", "Discussion & Limitations"),
]


def detect_academic_section(text: str) -> str:
    """Classify text chunk into a functional/academic section based on structural headings."""
    for pattern, section in SECTION_KEYWORDS:
        if re.search(pattern, text[:180]):
            return section
    return "Methodology & Context"


# ==============================================================================
# DOCUMENT EXTRACTORS FOR ALL SUPPORTED FORMATS
# ==============================================================================
def extract_text_from_pdf(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF document with title and metadata detection.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    reader = pypdf.PdfReader(io.BytesIO(data))
    doc_info = reader.metadata or {}
    raw_title = doc_info.get("/Title") or ""

    first_page_text = ""
    if len(reader.pages) > 0:
        first_page_text = reader.pages[0].extract_text() or ""

    if not raw_title or len(str(raw_title).strip()) < 4 or str(raw_title).lower().startswith("untitled"):
        lines = [line.strip() for line in first_page_text.split("\n") if len(line.strip()) > 5]
        paper_title = lines[0] if lines else os.path.splitext(fname)[0]
    else:
        paper_title = str(raw_title).strip()

    paper_title = re.sub(r"\s+", " ", paper_title)[:120]
    doc_id = compute_file_hash(data)

    extracted_pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = clean_text(text)
        if cleaned:
            extracted_pages.append({
                "doc_id": doc_id,
                "page_num": idx,
                "text": cleaned,
                "title": paper_title,
                "file_name": fname,
                "file_type": "pdf",
                "total_pages": len(reader.pages)
            })

    return extracted_pages


def extract_text_from_docx(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text, headings, and tables from a Word (.docx) document.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data or not HAS_DOCX:
        return []

    doc = docx.Document(io.BytesIO(data))
    doc_id = compute_file_hash(data)

    # Detect title from core properties or first heading
    doc_title = (doc.core_properties.title or "").strip()
    if not doc_title:
        for p in doc.paragraphs:
            t = p.text.strip()
            if t and (p.style.name.startswith("Heading") or len(t) > 5):
                doc_title = t
                break
    if not doc_title:
        doc_title = os.path.splitext(fname)[0]
    doc_title = re.sub(r"\s+", " ", doc_title)[:120]

    # Collect structured body elements (paragraphs & tables)
    body_blocks = []
    for p in doc.paragraphs:
        t = clean_text(p.text)
        if t:
            if p.style.name.startswith("Heading 1"):
                body_blocks.append(f"\n# {t}\n")
            elif p.style.name.startswith("Heading"):
                body_blocks.append(f"\n## {t}\n")
            else:
                body_blocks.append(t)

    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            if any(cells):
                table_rows.append(" | ".join(cells))
        if table_rows:
            body_blocks.append("\nTable Data:\n" + "\n".join(table_rows) + "\n")

    full_content = "\n\n".join(body_blocks)
    if not full_content.strip():
        return []

    # Simulate pages (~2,500 chars or 400 words per page)
    char_page_size = 2500
    pages = []
    paragraphs = full_content.split("\n\n")
    current_page_text = []
    current_len = 0
    page_counter = 1

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len > char_page_size and current_page_text:
            pages.append({
                "doc_id": doc_id,
                "page_num": page_counter,
                "text": "\n\n".join(current_page_text).strip(),
                "title": doc_title,
                "file_name": fname,
                "file_type": "docx",
            })
            page_counter += 1
            current_page_text = [p]
            current_len = p_len
        else:
            current_page_text.append(p)
            current_len += p_len

    if current_page_text:
        pages.append({
            "doc_id": doc_id,
            "page_num": page_counter,
            "text": "\n\n".join(current_page_text).strip(),
            "title": doc_title,
            "file_name": fname,
            "file_type": "docx",
        })

    for p in pages:
        p["total_pages"] = len(pages)

    return pages


def extract_text_from_doc(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Fallback extractor for legacy Word 97-2003 (.doc) binary files.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    doc_id = compute_file_hash(data)
    # Extract ASCII & UTF-8 printable string runs from binary stream
    strings = re.findall(rb"[\x20-\x7E\t\n\r]{4,}", data)
    extracted_text = " ".join(s.decode("latin-1", errors="ignore") for s in strings)
    cleaned = clean_text(extracted_text)
    if not cleaned:
        return []

    title = os.path.splitext(fname)[0]
    return [{
        "doc_id": doc_id,
        "page_num": 1,
        "text": cleaned,
        "title": title,
        "file_name": fname,
        "file_type": "doc",
        "total_pages": 1
    }]


def extract_text_from_txt(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from plain text (.txt) files with multi-encoding support.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    doc_id = compute_file_hash(data)
    decoded_text = ""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            decoded_text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    cleaned = clean_text(decoded_text)
    if not cleaned:
        return []

    # Title detection
    first_line = cleaned.split("\n")[0].strip()
    title = first_line[:100] if len(first_line) > 5 else os.path.splitext(fname)[0]

    # Split into simulated pages on form-feed or every 2,500 characters
    raw_pages = cleaned.split("\x0c") if "\x0c" in cleaned else []
    if not raw_pages or len(raw_pages) == 1:
        # Chunk into ~2,500 char sections
        paragraphs = cleaned.split("\n\n")
        raw_pages = []
        buf = []
        blen = 0
        for p in paragraphs:
            if blen + len(p) > 2500 and buf:
                raw_pages.append("\n\n".join(buf))
                buf = [p]
                blen = len(p)
            else:
                buf.append(p)
                blen += len(p)
        if buf:
            raw_pages.append("\n\n".join(buf))

    pages = []
    for idx, p_text in enumerate(raw_pages, start=1):
        pt = p_text.strip()
        if pt:
            pages.append({
                "doc_id": doc_id,
                "page_num": idx,
                "text": pt,
                "title": title,
                "file_name": fname,
                "file_type": "txt",
                "total_pages": len(raw_pages)
            })

    return pages


def extract_text_from_md(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text and sections from Markdown (.md) documents.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    doc_id = compute_file_hash(data)
    decoded_text = ""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            decoded_text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    cleaned = clean_text(decoded_text)
    if not cleaned:
        return []

    # Extract title from first H1 '# Title'
    title_match = re.search(r"^#\s+(.+)$", cleaned, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else os.path.splitext(fname)[0]

    # Split by major headings (# or ##) or ~2,500 chars
    sections = re.split(r"(?m)(?=^#{1,2}\s+)", cleaned)
    sections = [s.strip() for s in sections if s.strip()]

    pages = []
    for idx, sec in enumerate(sections, start=1):
        pages.append({
            "doc_id": doc_id,
            "page_num": idx,
            "text": sec,
            "title": title,
            "file_name": fname,
            "file_type": "md",
            "total_pages": len(sections)
        })

    return pages


def extract_text_from_csv(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract structured schema, distributions, and tabular samples from a CSV file.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    doc_id = compute_file_hash(data)
    title = f"Dataset: {os.path.splitext(fname)[0]}"

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        # Fallback to plain text
        return extract_text_from_txt(file_input, file_name)

    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes_summary = ", ".join([f"{c} ({t})" for c, t in zip(df.columns[:15], df.dtypes[:15])])

    # Page 1: Schema, Column Metadata, Missingness
    missing_info = df.isnull().sum()
    missing_str = ", ".join([f"{col}: {cnt} missing" for col, cnt in missing_info.items() if cnt > 0]) or "None"

    num_cols = list(df.select_dtypes(include="number").columns)
    stats_lines = []
    if num_cols:
        desc = df[num_cols[:8]].describe().round(3)
        stats_lines.append("Numerical Summary Statistics:\n" + desc.to_string())

    page1_text = (
        f"# Dataset Schema & Overview: {fname}\n\n"
        f"Total Records: {n_rows:,} rows | Total Features: {n_cols} columns\n"
        f"Columns: {', '.join(cols[:25])}\n"
        f"Data Types: {dtypes_summary}\n"
        f"Missing Values: {missing_str}\n\n"
        + "\n\n".join(stats_lines)
    )

    pages = [{
        "doc_id": doc_id,
        "page_num": 1,
        "text": page1_text,
        "title": title,
        "file_name": fname,
        "file_type": "csv",
        "section": "Data Schema & Diagnostics"
    }]

    # Page 2: Sample Observations
    sample_df = df.head(15)
    sample_str = sample_df.to_string(index=False)
    page2_text = (
        f"# Sample Observations: {fname}\n\n"
        f"First {len(sample_df)} sample records:\n"
        f"```\n{sample_str}\n```"
    )
    pages.append({
        "doc_id": doc_id,
        "page_num": 2,
        "text": page2_text,
        "title": title,
        "file_name": fname,
        "file_type": "csv",
        "section": "Sample Records"
    })

    for p in pages:
        p["total_pages"] = len(pages)

    return pages


def extract_text_from_excel(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract multi-sheet tables, summaries, and schemas from Excel (.xlsx, .xls) workbooks.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data:
        return []

    doc_id = compute_file_hash(data)
    title = f"Workbook: {os.path.splitext(fname)[0]}"

    try:
        excel_file = pd.ExcelFile(io.BytesIO(data))
    except Exception as exc:
        return []

    sheet_names = excel_file.sheet_names
    pages = []

    for idx, sname in enumerate(sheet_names, start=1):
        try:
            sdf = pd.read_excel(excel_file, sheet_name=sname)
        except Exception:
            continue

        n_r, n_c = sdf.shape
        cols = list(sdf.columns)
        num_cols = list(sdf.select_dtypes(include="number").columns)
        stats_str = ""
        if num_cols:
            stats_str = "\nSummary Statistics:\n" + sdf[num_cols[:6]].describe().round(2).to_string()

        sample_str = sdf.head(10).to_string(index=False) if not sdf.empty else "Empty Sheet"
        sheet_text = (
            f"# Sheet {idx}: {sname} ({fname})\n\n"
            f"Dimensions: {n_r:,} rows x {n_c} columns\n"
            f"Columns: {', '.join(str(c) for c in cols[:20])}\n"
            f"{stats_str}\n\n"
            f"Sample Data:\n```\n{sample_str}\n```"
        )

        pages.append({
            "doc_id": doc_id,
            "page_num": idx,
            "text": sheet_text,
            "title": title,
            "file_name": fname,
            "file_type": "xlsx",
            "section": f"Sheet: {sname}"
        })

    for p in pages:
        p["total_pages"] = len(pages)

    return pages


def extract_text_from_pptx(file_input: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract slide presentation content, shapes, and notes from PowerPoint (.pptx).
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    if not data or not HAS_PPTX:
        return []

    doc_id = compute_file_hash(data)
    prs = pptx.Presentation(io.BytesIO(data))
    title = os.path.splitext(fname)[0]

    pages = []
    for s_idx, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        slide_title = ""

        # Extract shapes & text frames
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = clean_text(para.text)
                    if t:
                        if shape == slide.shapes.title and not slide_title:
                            slide_title = t
                        else:
                            slide_texts.append(t)
            elif shape.has_table:
                table_lines = []
                for row in shape.table.rows:
                    cells = [clean_text(cell.text) for cell in row.cells]
                    if any(cells):
                        table_lines.append(" | ".join(cells))
                if table_lines:
                    slide_texts.append("\nTable:\n" + "\n".join(table_lines))

        # Check speaker notes
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = clean_text(slide.notes_slide.notes_text_frame.text)
            if notes:
                slide_texts.append(f"Speaker Notes: {notes}")

        stitle_display = slide_title or f"Slide {s_idx}"
        slide_content = f"# Slide {s_idx}: {stitle_display}\n\n" + "\n\n".join(slide_texts)
        if slide_content.strip():
            pages.append({
                "doc_id": doc_id,
                "page_num": s_idx,
                "text": slide_content,
                "title": title,
                "file_name": fname,
                "file_type": "pptx",
                "section": f"Slide {s_idx}: {stitle_display[:40]}"
            })

    for p in pages:
        p["total_pages"] = len(pages)

    return pages


def extract_document_pages(file_input: Any, file_name: str) -> Dict[str, Any]:
    """
    Unified extraction dispatcher: Detects file format by extension and routes
    to appropriate extractor for PDF, DOCX, DOC, TXT, MD, CSV, XLSX, XLS, PPTX.
    """
    data, fname = read_input_bytes(file_input, default_name=file_name)
    fname = file_name or fname
    ext = os.path.splitext(fname)[1].lower()
    doc_id = compute_file_hash(data) if data else "empty"

    if ext == ".pdf":
        pages = extract_text_from_pdf(data, fname)
        ftype = "pdf"
    elif ext == ".docx":
        pages = extract_text_from_docx(data, fname)
        ftype = "docx"
    elif ext == ".doc":
        pages = extract_text_from_doc(data, fname)
        ftype = "doc"
    elif ext in [".txt", ".log"]:
        pages = extract_text_from_txt(data, fname)
        ftype = "txt"
    elif ext == ".md":
        pages = extract_text_from_md(data, fname)
        ftype = "md"
    elif ext == ".csv":
        pages = extract_text_from_csv(data, fname)
        ftype = "csv"
    elif ext in [".xlsx", ".xls"]:
        pages = extract_text_from_excel(data, fname)
        ftype = "xlsx"
    elif ext == ".pptx":
        pages = extract_text_from_pptx(data, fname)
        ftype = "pptx"
    else:
        # Fallback to plain text
        pages = extract_text_from_txt(data, fname)
        ftype = "txt"

    title = pages[0]["title"] if pages else os.path.splitext(fname)[0]
    return {
        "doc_id": doc_id,
        "title": title,
        "file_name": fname,
        "file_type": ftype,
        "pages": pages,
        "total_pages": len(pages),
        "raw_bytes_len": len(data)
    }


# ==============================================================================
# INTELLIGENT TEXT CHUNKING WITH OVERLAP CONFIGURATION
# ==============================================================================
def intelligent_chunk_document(
    pages: List[Dict[str, Any]],
    target_chunk_size: int = 700,
    overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Chunk extracted document pages into semantically cohesive passages while preserving
    exact page numbers, document title, file type, sentence boundaries, and functional sections.
    """
    chunks = []
    chunk_counter = 0

    for page_info in pages:
        page_num = page_info["page_num"]
        title = page_info["title"]
        file_name = page_info["file_name"]
        file_type = page_info.get("file_type", os.path.splitext(file_name)[1].lstrip(".").lower() or "doc")
        doc_id = page_info.get("doc_id", compute_file_hash(file_name.encode("utf-8")))
        text = page_info["text"]

        # Split into paragraphs and sentences
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_len = 0
        current_section = page_info.get("section", "Methodology & Context")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            sec = detect_academic_section(para)
            if sec != "Methodology & Context":
                current_section = sec

            # Split paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_len = len(sentence)
                if current_len + sentence_len > target_chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                        "doc_id": doc_id,
                        "title": title,
                        "file_name": file_name,
                        "file_type": file_type,
                        "page_num": page_num,
                        "section": current_section,
                        "text": chunk_text,
                        "char_count": len(chunk_text),
                        "upload_timestamp": datetime.now(timezone.utc).isoformat()
                    })

                    # Maintain overlap from trailing sentences
                    overlap_chunk = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) <= overlap:
                            overlap_chunk.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break

                    current_chunk = overlap_chunk
                    current_len = overlap_len

                current_chunk.append(sentence)
                current_len += sentence_len

        # Flush trailing chunk on page
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_c{chunk_counter}",
                "doc_id": doc_id,
                "title": title,
                "file_name": file_name,
                "file_type": file_type,
                "page_num": page_num,
                "section": current_section,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "upload_timestamp": datetime.now(timezone.utc).isoformat()
            })

    return chunks


# ==============================================================================
# BM25 LEXICAL INDEX & HYBRID RANK FUSION (RRF)
# ==============================================================================
class BM25Index:
    """
    Lightweight in-memory BM25Okapi lexical index.
    Complements dense embeddings for Hybrid Semantic + Exact Lexical Retrieval.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.num_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text) if len(w) > 1]

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

            self.doc_term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            self.docs.append(doc)
            self.num_docs += 1

        if self.num_docs > 0:
            self.avg_doc_len = sum(self.doc_lens) / float(self.num_docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.num_docs == 0:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = []
        for i, doc in enumerate(self.docs):
            if paper_filter and doc.get("title") != paper_filter and doc.get("file_name") != paper_filter:
                continue
            if file_type_filter and doc.get("file_type") != file_type_filter:
                continue

            score = 0.0
            doc_len = self.doc_lens[i]
            tf_map = self.doc_term_freqs[i]

            for term in q_tokens:
                if term in tf_map:
                    tf = tf_map[term]
                    df = self.doc_freqs.get(term, 0)
                    idf = float(np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0))
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += idf * (num / (den or 1.0))

            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, doc in scores[:top_k]:
            item = dict(doc)
            item["lexical_score"] = float(round(s, 4))
            results.append(item)
        return results

    def clear(self):
        self.docs = []
        self.doc_lens = []
        self.avg_doc_len = 0.0
        self.doc_freqs = {}
        self.doc_term_freqs = []
        self.num_docs = 0


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 5,
    dense_weight: float = 0.6,
    lexical_weight: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Combine dense semantic retrieval and BM25 lexical retrieval using Reciprocal Rank Fusion.
    Guarantees robust balance between semantic intent and exact causal formula/parameter matches.
    """
    scores: Dict[str, float] = {}
    passages: Dict[str, Dict[str, Any]] = {}

    for rank, p in enumerate(dense_results):
        key = f"{p['file_name']}_p{p['page_num']}_{p['text'][:60]}"
        scores[key] = scores.get(key, 0.0) + dense_weight / (k + rank + 1)
        if key not in passages:
            passages[key] = dict(p)
            passages[key]["dense_rank"] = rank + 1
        else:
            passages[key]["dense_rank"] = rank + 1

    for rank, p in enumerate(lexical_results):
        key = f"{p['file_name']}_p{p['page_num']}_{p['text'][:60]}"
        scores[key] = scores.get(key, 0.0) + lexical_weight / (k + rank + 1)
        if key not in passages:
            passages[key] = dict(p)
            passages[key]["lexical_rank"] = rank + 1
        else:
            passages[key]["lexical_rank"] = rank + 1

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_results = []

    for key in sorted_keys[:top_n]:
        item = passages[key]
        item["rrf_score"] = float(round(scores[key], 5))
        if "similarity_score" not in item:
            item["similarity_score"] = float(round(item.get("lexical_score", 1.0) / 10.0, 4))
        fused_results.append(item)

    return fused_results


# ==============================================================================
# PERSISTENT CHROMADB MULTI-DOCUMENT VECTOR STORE
# ==============================================================================
class AcademicVectorStore:
    """
    ChromaDB-backed vector store supporting multi-format document ingestion
    (PDF, DOCX, DOC, TXT, MD, CSV, XLSX, PPTX), metadata tracking, deduplication,
    BM25 lexical search, and Hybrid RRF retrieval.
    """

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "causal_literature"):
        if persist_directory is None:
            persist_directory = os.getenv("CHROMADB_PERSIST_DIR", "data/chroma_db")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index = BM25Index()
        self._sync_bm25_from_chroma()

    def _sync_bm25_from_chroma(self):
        """Populate the in-memory BM25 index from existing records in ChromaDB."""
        try:
            records = self.collection.get()
            if records and "documents" in records and records["documents"]:
                docs = []
                for d, m in zip(records["documents"], records["metadatas"]):
                    docs.append({
                        "doc_id": m.get("doc_id", "doc"),
                        "title": m.get("title", "Untitled"),
                        "file_name": m.get("file_name", "document"),
                        "file_type": m.get("file_type", "pdf"),
                        "page_num": int(m.get("page_num", 1)),
                        "section": m.get("section", "Methodology & Context"),
                        "text": d
                    })
                self.bm25_index.clear()
                self.bm25_index.add_documents(docs)
        except Exception:
            pass

    def _get_embedding_vector(self, text: str, api_key: Any = _ENV_DEFAULT) -> List[float]:
        """
        Generate embedding vector via Google Gemini API when API key is available,
        or fall back to deterministic semantic subword frequency vector for offline testing.
        """
        if api_key is _ENV_DEFAULT:
            key = get_api_key()
        else:
            key = api_key

        # Check existing collection dimensionality if collection has records
        target_dim = None
        if self.collection.count() > 0:
            try:
                first = self.collection.get(limit=1, include=["embeddings"])
                if first and first.get("embeddings") and len(first["embeddings"]) > 0:
                    target_dim = len(first["embeddings"][0])
            except Exception:
                pass

        if key:
            try:
                emb = generate_gemini_embedding(text[:3000], api_key=key)
                if emb:
                    if target_dim is None or len(emb) == target_dim:
                        return emb
            except Exception:
                pass

        # Deterministic local semantic feature vector (normalized to target_dim or 128)
        dim = target_dim if target_dim else 128
        vec = np.zeros(dim)
        words = re.findall(r"\w+", text.lower())
        for w in words:
            h = hash(w) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def is_document_indexed(self, doc_id: str, file_name: Optional[str] = None) -> bool:
        """
        Check if a document fingerprint (doc_id) or exact file_name has already been indexed.
        Prevents redundant re-processing.
        """
        if self.collection.count() == 0:
            return False

        try:
            records = self.collection.get(where={"doc_id": doc_id}, limit=1)
            if records and records.get("ids"):
                return True
        except Exception:
            pass

        if file_name:
            try:
                records = self.collection.get(where={"file_name": file_name}, limit=1)
                if records and records.get("ids"):
                    return True
            except Exception:
                pass

        return False

    def ingest_document(
        self,
        file_input: Any,
        file_name: Optional[str] = None,
        api_key: Optional[str] = None,
        target_chunk_size: int = 700,
        overlap: int = 120,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """Convenience method to ingest a single document from bytes, path, or file-like object."""
        if isinstance(file_input, bytes):
            buf = io.BytesIO(file_input)
            buf.name = file_name or "document"
            file_obj = buf
        else:
            file_obj = file_input
            if file_name and not hasattr(file_obj, "name"):
                setattr(file_obj, "name", file_name)
        res = self.add_documents(
            [file_obj],
            api_key=api_key,
            target_chunk_size=target_chunk_size,
            overlap=overlap,
            force_reindex=force_reindex
        )
        return {
            "success": len(res.get("indexed_files", [])) > 0 or len(res.get("skipped_files", [])) > 0,
            "result": res
        }

    def add_documents(
        self,
        uploaded_files: List[Any],
        api_key: Optional[str] = None,
        target_chunk_size: int = 700,
        overlap: int = 120,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Process, validate, extract, chunk, embed, and index multiple documents
        (PDF, Word, Text, Markdown, CSV, Excel, PowerPoint) into ChromaDB.
        Skips already-indexed files unless force_reindex is True.
        """
        total_pages = 0
        total_chunks = 0
        indexed_titles = []
        indexed_files = []
        skipped_files = []
        errors = []

        for f in uploaded_files:
            file_name = getattr(f, "name", "document")
            try:
                extracted = extract_document_pages(f, file_name)
            except Exception as exc:
                errors.append({"file_name": file_name, "error": str(exc)})
                continue

            doc_id = extracted["doc_id"]
            pages = extracted["pages"]
            doc_title = extracted["title"]
            file_type = extracted["file_type"]

            if not pages:
                errors.append({"file_name": file_name, "error": "No readable content extracted"})
                continue

            # Check if document has already been indexed
            if not force_reindex and self.is_document_indexed(doc_id, file_name):
                skipped_files.append({
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "title": doc_title,
                    "reason": "Already indexed (duplicate avoided)"
                })
                continue

            # If force_reindex, purge existing chunks for this doc first
            if force_reindex and self.is_document_indexed(doc_id, file_name):
                self.delete_document(doc_id)

            total_pages += len(pages)
            indexed_titles.append(doc_title)
            indexed_files.append(file_name)

            chunks = intelligent_chunk_document(
                pages=pages,
                target_chunk_size=target_chunk_size,
                overlap=overlap
            )
            if not chunks:
                continue

            ids = []
            docs = []
            metadatas = []
            embeddings = []

            for c in chunks:
                ids.append(c["chunk_id"])
                docs.append(c["text"])
                metadatas.append({
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "file_name": c["file_name"],
                    "file_type": c.get("file_type", file_type),
                    "page_num": int(c["page_num"]),
                    "section": c.get("section", "Methodology & Context"),
                    "char_count": int(c["char_count"]),
                    "upload_timestamp": c.get("upload_timestamp", datetime.now(timezone.utc).isoformat())
                })
                embeddings.append(self._get_embedding_vector(c["text"], api_key))

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metadatas,
                embeddings=embeddings
            )
            total_chunks += len(chunks)

        self._sync_bm25_from_chroma()

        return {
            "documents_indexed": len(indexed_files),
            "documents_skipped": len(skipped_files),
            "indexed_files": indexed_files,
            "skipped_files": skipped_files,
            "titles": list(set(indexed_titles)),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "collection_count": self.collection.count(),
            "errors": errors,
            # Backwards compatibility keys
            "papers_indexed": len(set(indexed_titles)),
        }

    def add_pdf_documents(self, uploaded_files: List[Any], api_key: Optional[str] = None) -> Dict[str, Any]:
        """Backward-compatible wrapper for add_documents."""
        return self.add_documents(uploaded_files, api_key=api_key, force_reindex=True)

    def delete_document(self, doc_identifier: str) -> bool:
        """
        Delete all chunks associated with a specific doc_id or file_name.
        """
        try:
            records = self.collection.get()
            if not records or "metadatas" not in records or not records["metadatas"]:
                return False

            ids_to_delete = []
            for cid, meta in zip(records["ids"], records["metadatas"]):
                if (meta.get("doc_id") == doc_identifier or
                    meta.get("file_name") == doc_identifier or
                    meta.get("title") == doc_identifier):
                    ids_to_delete.append(cid)

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self._sync_bm25_from_chroma()
                return True
        except Exception:
            pass
        return False

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        paper_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        min_similarity: float = 0.0,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Retrieval combining Dense Cosine Semantic Search and
        BM25Okapi Lexical Matching via Reciprocal Rank Fusion (RRF).
        Supports filtering by document title, filename, file_type, and minimum similarity threshold.
        """
        count = self.collection.count()
        if count == 0:
            return []

        # Ensure BM25 lexical index is synchronized with Chroma
        if self.bm25_index.num_docs == 0 and count > 0:
            self._sync_bm25_from_chroma()

        # 1. Dense Semantic Query
        query_embedding = self._get_embedding_vector(query, api_key)
        where_conditions = []
        if paper_filter and paper_filter != "All Documents" and paper_filter != "All Indexed Papers":
            where_conditions.append({"title": paper_filter})
        if file_type_filter and file_type_filter != "All Formats":
            where_conditions.append({"file_type": file_type_filter.lower()})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        dense_passages = []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max(top_k * 2, 4), count),
                where=where_filter
            )
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

                for d, m, dist in zip(docs, metas, dists):
                    similarity = round(float(max(0.0, 1.0 - dist)), 4) if dist is not None else 1.0
                    dense_passages.append({
                        "doc_id": m.get("doc_id", "doc"),
                        "title": m.get("title", "Unknown"),
                        "file_name": m.get("file_name", "Unknown"),
                        "file_type": m.get("file_type", "pdf"),
                        "page_num": int(m.get("page_num", 1)),
                        "section": m.get("section", "Methodology & Context"),
                        "text": d,
                        "similarity_score": similarity,
                    })
        except Exception:
            pass

        # 2. BM25 Lexical Query
        clean_pfilter = None if paper_filter in ["All Documents", "All Indexed Papers"] else paper_filter
        clean_tfilter = None if file_type_filter in ["All Formats", None] else file_type_filter.lower()
        lexical_passages = self.bm25_index.search(
            query=query,
            top_k=max(top_k * 2, 4),
            paper_filter=clean_pfilter,
            file_type_filter=clean_tfilter
        )

        # 3. Hybrid Fusion via Reciprocal Rank Fusion (RRF)
        fused_results = []
        if dense_passages and lexical_passages:
            fused = reciprocal_rank_fusion(dense_passages, lexical_passages, top_n=top_k)
            fused_results = fused if fused else dense_passages[:top_k]
        elif dense_passages:
            fused_results = dense_passages[:top_k]
        elif lexical_passages:
            for item in lexical_passages:
                if "similarity_score" not in item:
                    item["similarity_score"] = 0.85
            fused_results = lexical_passages[:top_k]

        # Apply minimum similarity filter if requested
        if min_similarity > 0.0 and fused_results:
            fused_results = [p for p in fused_results if float(p.get("similarity_score", 1.0)) >= min_similarity]

        return fused_results

    def search_literature(
        self,
        query: str,
        top_k: int = 5,
        paper_filter: Optional[str] = None,
        min_similarity: float = 0.0,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for search_documents."""
        return self.search_documents(
            query=query,
            top_k=top_k,
            paper_filter=paper_filter,
            min_similarity=min_similarity,
            api_key=api_key
        )
        """Backward-compatible wrapper for search_documents."""
        return self.search_documents(query=query, top_k=top_k, paper_filter=paper_filter, api_key=api_key)

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        List all distinct documents across all formats currently indexed in the vector store.
        """
        count = self.collection.count()
        if count == 0:
            return []

        all_records = self.collection.get()
        metas = all_records.get("metadatas", [])
        doc_stats = {}

        for m in metas:
            doc_id = m.get("doc_id", m.get("file_name", "doc"))
            title = m.get("title", "Untitled")
            fname = m.get("file_name", "document")
            ftype = m.get("file_type", os.path.splitext(fname)[1].lstrip(".").lower() or "doc")
            page = m.get("page_num", 1)
            char_count = int(m.get("char_count", 0))

            if doc_id not in doc_stats:
                doc_stats[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "file_name": fname,
                    "file_type": ftype,
                    "pages": set(),
                    "chunks": 0,
                    "total_chars": 0,
                    "upload_timestamp": m.get("upload_timestamp", m.get("indexed_at", "N/A"))
                }
            doc_stats[doc_id]["pages"].add(page)
            doc_stats[doc_id]["chunks"] += 1
            doc_stats[doc_id]["total_chars"] += char_count

        summary = []
        for p in doc_stats.values():
            summary.append({
                "doc_id": p["doc_id"],
                "title": p["title"],
                "file_name": p["file_name"],
                "file_type": p["file_type"],
                "total_pages": len(p["pages"]),
                "total_chunks": p["chunks"],
                "total_chars": p["total_chars"],
                "upload_timestamp": p["upload_timestamp"],
                # Backward-compatibility key
                "indexed_at": p["upload_timestamp"]
            })
        return summary

    def list_indexed_papers(self) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for list_indexed_documents."""
        return self.list_indexed_documents()

    def get_document_preview(self, doc_identifier: str, limit_chunks: int = 5) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed document metadata, structural stats, and a preview of extracted passage chunks.
        Matches doc_identifier against doc_id, file_name, or title.
        """
        count = self.collection.count()
        if count == 0:
            return None

        records = self.collection.get()
        if not records or "documents" not in records or not records["documents"]:
            return None

        matching_chunks = []
        matching_meta = None

        for cid, doc_text, meta in zip(records["ids"], records["documents"], records["metadatas"]):
            if (meta.get("doc_id") == doc_identifier or
                meta.get("file_name") == doc_identifier or
                meta.get("title") == doc_identifier):
                if matching_meta is None:
                    matching_meta = meta
                matching_chunks.append({
                    "chunk_id": cid,
                    "page_num": int(meta.get("page_num", 1)),
                    "section": meta.get("section", "Context"),
                    "char_count": int(meta.get("char_count", len(doc_text))),
                    "text": doc_text
                })

        if not matching_meta or not matching_chunks:
            return None

        # Sort chunks by page_num, then chunk_id
        matching_chunks.sort(key=lambda x: (x["page_num"], x["chunk_id"]))
        distinct_pages = set(c["page_num"] for c in matching_chunks)
        total_chars = sum(c["char_count"] for c in matching_chunks)

        return {
            "doc_id": matching_meta.get("doc_id", "doc"),
            "title": matching_meta.get("title", "Untitled"),
            "file_name": matching_meta.get("file_name", "document"),
            "file_type": matching_meta.get("file_type", "pdf"),
            "total_pages": len(distinct_pages),
            "total_chunks": len(matching_chunks),
            "total_chars": total_chars,
            "upload_timestamp": matching_meta.get("upload_timestamp", "N/A"),
            "chunks_preview": matching_chunks[:limit_chunks],
            "all_chunks_count": len(matching_chunks)
        }

    def reindex_document(
        self,
        doc_identifier: str,
        target_chunk_size: int = 700,
        overlap: int = 120,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Re-index an existing document in ChromaDB with updated chunking and overlap parameters.
        Reconstructs document structural pages from stored chunks, purges prior vectors,
        re-chunks, and embeds newly configured passages into ChromaDB and BM25 index.
        """
        count = self.collection.count()
        if count == 0:
            return {"success": False, "error": "Vector store is empty."}

        records = self.collection.get()
        if not records or "documents" not in records or not records["documents"]:
            return {"success": False, "error": "No records found in vector store."}

        doc_meta = None
        ids_to_purge = []
        page_dict: Dict[int, List[str]] = {}

        for cid, doc_text, meta in zip(records["ids"], records["documents"], records["metadatas"]):
            if (meta.get("doc_id") == doc_identifier or
                meta.get("file_name") == doc_identifier or
                meta.get("title") == doc_identifier):
                if doc_meta is None:
                    doc_meta = meta
                ids_to_purge.append(cid)
                p_num = int(meta.get("page_num", 1))
                if p_num not in page_dict:
                    page_dict[p_num] = []
                page_dict[p_num].append(doc_text)

        if not doc_meta or not ids_to_purge:
            return {"success": False, "error": f"Document '{doc_identifier}' not found in index."}

        doc_id = doc_meta.get("doc_id", "doc")
        title = doc_meta.get("title", "Untitled")
        file_name = doc_meta.get("file_name", "document")
        file_type = doc_meta.get("file_type", "pdf")

        # Reconstruct pages list
        pages = []
        for p_num in sorted(page_dict.keys()):
            # Combine chunk text for each structural page/sheet/slide
            combined_page_text = " ".join(page_dict[p_num])
            pages.append({
                "doc_id": doc_id,
                "title": title,
                "file_name": file_name,
                "file_type": file_type,
                "page_num": p_num,
                "text": combined_page_text
            })

        # Purge existing vectors for this document
        self.collection.delete(ids=ids_to_purge)

        # Re-chunk with updated parameters
        new_chunks = intelligent_chunk_document(
            pages=pages,
            target_chunk_size=target_chunk_size,
            overlap=overlap
        )
        if not new_chunks:
            self._sync_bm25_from_chroma()
            return {"success": True, "doc_id": doc_id, "new_chunks_count": 0, "file_name": file_name}

        new_ids = []
        new_docs = []
        new_metas = []
        new_embs = []

        now_ts = datetime.now(timezone.utc).isoformat()
        for c in new_chunks:
            new_ids.append(c["chunk_id"])
            new_docs.append(c["text"])
            new_metas.append({
                "doc_id": c["doc_id"],
                "title": c["title"],
                "file_name": c["file_name"],
                "file_type": c.get("file_type", file_type),
                "page_num": int(c["page_num"]),
                "section": c.get("section", "Methodology & Context"),
                "char_count": int(c["char_count"]),
                "upload_timestamp": now_ts
            })
            new_embs.append(self._get_embedding_vector(c["text"], api_key))

        self.collection.upsert(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_metas,
            embeddings=new_embs
        )
        self._sync_bm25_from_chroma()

        return {
            "success": True,
            "doc_id": doc_id,
            "title": title,
            "file_name": file_name,
            "previous_chunks_count": len(ids_to_purge),
            "new_chunks_count": len(new_chunks),
            "total_collection_count": self.collection.count()
        }

    def reindex_all_documents(
        self,
        target_chunk_size: int = 700,
        overlap: int = 120,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Re-index all documents currently stored in the vector store with updated chunk size and overlap.
        """
        docs = self.list_indexed_documents()
        reindexed = []
        for d in docs:
            res = self.reindex_document(
                doc_identifier=d["doc_id"],
                target_chunk_size=target_chunk_size,
                overlap=overlap,
                api_key=api_key
            )
            reindexed.append(res)
        return {
            "total_documents_reindexed": len(reindexed),
            "details": reindexed,
            "collection_count": self.collection.count()
        }

    def clear_store(self):
        """Reset and clear all indexed documents from ChromaDB and BM25 index."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index.clear()

    def seed_benchmark_papers(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Seed foundational peer-reviewed causal inference literature directly into ChromaDB.
        """
        papers_data = [
            {
                "title": "Double/Debiased Machine Learning for Treatment and Structural Parameters (Chernozhukov et al., 2018)",
                "file_name": "chernozhukov_2018_doubleml.pdf",
                "file_type": "pdf",
                "doc_id": "seed_chernozhukov_2018",
                "chunks": [
                    {
                        "page_num": 1,
                        "section": "Methodology & Estimators",
                        "text": "Double Machine Learning (DML) provides root-N consistent and asymptotically normal estimators of low-dimensional structural parameters in the presence of high-dimensional nuisance parameters estimated via arbitrary machine learning algorithms (Random Forests, Gradient Boosting, Lasso, Neural Networks). Standard ML methods induce severe regularization bias in causal effect estimates due to shrinkage or overfitting. DML solves this via two foundational mechanisms: Neyman-orthogonal score equations and sample-splitting (cross-fitting)."
                    },
                    {
                        "page_num": 12,
                        "section": "Identification & Assumptions",
                        "text": "Neyman orthogonality ensures that the first-order derivative of the moment condition with respect to nuisance functions is zero at the true parameter values. This eliminates first-order sensitivity to nuisance estimation errors. Sample splitting (K-fold cross-fitting) breaks the dependence between the main estimation step and the nuisance parameter estimation step, preventing overfitting and guaranteeing that the empirical process term remains asymptotically negligible."
                    }
                ]
            },
            {
                "title": "Sensitivity Analysis in Observational Research: Introducing the E-Value (VanderWeele & Ding, 2017)",
                "file_name": "vanderweele_2017_evalue.pdf",
                "file_type": "pdf",
                "doc_id": "seed_vanderweele_2017",
                "chunks": [
                    {
                        "page_num": 3,
                        "section": "Sensitivity & Robustness",
                        "text": "The E-value is defined as the minimum strength of association, on the risk ratio scale, that an unmeasured confounder would need to have with both the treatment and the outcome, conditional on the measured covariates, to fully explain away a specific treatment-outcome association. Unlike traditional sensitivity analyses that require specifying multiple sensitivity parameters (prevalence, outcome association, treatment association), the E-value reduces sensitivity to a single, intuitive, and readily interpretable metric."
                    },
                    {
                        "page_num": 8,
                        "section": "Sensitivity & Robustness",
                        "text": "We recommend reporting the E-value for the point estimate as well as the E-value for the confidence interval limit closest to the null hypothesis. If the lower confidence limit's E-value exceeds 1.5 or 2.0, an unmeasured confounder would need to be exceptionally strong (stronger than most established clinical risk factors) to reduce the effect estimate to statistical insignificance. For continuous outcomes analyzed via standardized effect sizes, Cohen's d is converted to an approximate risk ratio via the transformation RR = exp(0.91 * d)."
                    }
                ]
            },
            {
                "title": "Estimation with Missing Regressors & Doubly Robust AIPW (Robins, Rotnitzky, & Zhao, 1994)",
                "file_name": "robins_1994_aipw.pdf",
                "file_type": "pdf",
                "doc_id": "seed_robins_1994",
                "chunks": [
                    {
                        "page_num": 4,
                        "section": "Methodology & Estimators",
                        "text": "Augmented Inverse Probability Weighting (AIPW) combines a propensity score model e(X) = P(D=1|X) and an outcome regression model mu(d, X) = E[Y|D=d, X]. The estimator is doubly robust: it yields a consistent and asymptotically normal estimate of the average treatment effect (ATE) if EITHER the propensity score model is correctly specified OR the outcome regression model is correctly specified, but not necessarily both."
                    },
                    {
                        "page_num": 15,
                        "section": "Methodology & Estimators",
                        "text": "Furthermore, when both the propensity score and outcome models are correctly specified, the AIPW estimator achieves the semi-parametric efficiency bound for the average treatment effect. Standard IPW estimators suffer from extreme weights when propensity scores approach 0 or 1 (positivity violations), causing high variance. AIPW incorporates the outcome regression as a control variate, drastically reducing variance and bias even under moderate overlap issues."
                    }
                ]
            }
        ]

        ids = []
        docs = []
        metadatas = []
        embeddings = []
        indexed_titles = []
        total_chunks = 0

        for p_idx, paper in enumerate(papers_data):
            indexed_titles.append(paper["title"])
            for c_idx, c in enumerate(paper["chunks"]):
                cid = f"seed_{p_idx+1}_c{c_idx+1}"
                text = c["text"]
                ids.append(cid)
                docs.append(text)
                metadatas.append({
                    "doc_id": paper["doc_id"],
                    "title": paper["title"],
                    "file_name": paper["file_name"],
                    "file_type": paper["file_type"],
                    "page_num": c["page_num"],
                    "section": c.get("section", "Methodology & Context"),
                    "char_count": len(text),
                    "upload_timestamp": datetime.now(timezone.utc).isoformat()
                })
                embeddings.append(self._get_embedding_vector(text, api_key))
                total_chunks += 1

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings
        )

        self._sync_bm25_from_chroma()

        return {
            "papers_indexed": len(indexed_titles),
            "documents_indexed": len(indexed_titles),
            "titles": indexed_titles,
            "total_chunks": total_chunks,
            "collection_count": self.collection.count()
        }


# ==============================================================================
# RAG QUESTION ANSWERING & GROUNDED EVIDENCE SYNTHESIS
# ==============================================================================
def answer_document_query(
    query: str,
    vector_store: AcademicVectorStore,
    api_key: Any = _ENV_DEFAULT,
    model_name: Optional[str] = None,
    top_k: int = 5,
    paper_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None,
    min_similarity: float = 0.0
) -> Dict[str, Any]:
    """
    RAG Multi-Document Question Answering: Retrieve top semantic passages from
    uploaded documents (PDF, Word, Excel, CSV, PPTX, MD, TXT) and answer user
    queries with strictly grounded citations and evidence coordinates.
    """
    passages = vector_store.search_documents(
        query=query,
        top_k=top_k,
        paper_filter=paper_filter,
        file_type_filter=file_type_filter,
        min_similarity=min_similarity,
        api_key=api_key
    )
    if not passages:
        return {
            "answer": "The ingested documents do not contain sufficient evidence to address this question.",
            "citations": [],
            "retrieved_passages": [],
            "insufficient_evidence": True
        }

    # Format passages as numbered evidence blocks
    passage_blocks = []
    for i, p in enumerate(passages, start=1):
        ftype_label = p.get("file_type", "doc").upper()
        passage_blocks.append(
            f"[Source {i}] Document: \"{p['title']}\" | Format: {ftype_label} | Page/Slide/Sheet: {p['page_num']} | File: {p['file_name']} | Section: {p.get('section', 'Context')} | Similarity: {p.get('similarity_score', 1.0):.1%}\n"
            f"Passage Excerpt: \"{p['text']}\"\n"
        )
    context_text = "\n".join(passage_blocks)

    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    if not key:
        # Fallback offline synthesis
        fallback_lines = [
            "**Notice:** Google Gemini API key is not configured. Displaying top retrieved passages directly from ChromaDB & BM25:\n"
        ]
        citations = []
        for i, p in enumerate(passages, 1):
            ft = p.get("file_type", "doc").upper()
            fallback_lines.append(f"**[{i}] [{ft}] {p['title']} (Page/Unit {p['page_num']})** — Similarity: `{p['similarity_score']:.1%}` | Section: `{p.get('section', 'General')}`")
            fallback_lines.append(f"> {p['text'][:320]}...\n")
            citations.append({
                "paper": p["title"],
                "title": p["title"],
                "file": p["file_name"],
                "file_name": p["file_name"],
                "file_type": p.get("file_type", "doc"),
                "page": p["page_num"],
                "page_num": p["page_num"],
                "section": p.get("section", "Context"),
                "excerpt": p["text"][:220],
                "similarity_score": p["similarity_score"]
            })

        return {
            "answer": "\n".join(fallback_lines),
            "citations": citations,
            "retrieved_passages": passages,
            "insufficient_evidence": False
        }

    system_prompt = """
You are CausalLens AI's Document Intelligence and RAG Research Assistant.
Answer the user's question ONLY using the provided verified document excerpts below.

CRITICAL EVIDENCE & CITATION RULES:
1. EVERY assertion, fact, parameter, or finding MUST explicitly cite its source document title, format, and page/slide/sheet number using:
   [Source N: "<Title>", Type: <FORMAT>, Page/Unit: <PageNum>]
2. NEVER FABRICATE CITATIONS, DATASETS, EXPERIMENTAL FINDINGS, OR EXTERNAL REFERENCES.
   Cite ONLY the exact passages provided in the context below.
3. If the provided excerpts do not contain sufficient evidence to answer the question, clearly state:
   "The ingested documents do not contain sufficient evidence to address this question."
4. Clearly communicate any limitations, data quality constraints, or unmeasured assumptions disclosed in the documents.
"""

    prompt = f"""
{system_prompt}

INGESTED DOCUMENT EXCERPTS:
{context_text}

USER RESEARCH QUESTION:
{query}
"""

    gen_res = generate_gemini_text(
        prompt=prompt,
        system_instruction=system_prompt,
        preferred_model=model_name,
        api_key=key,
    )

    if gen_res["success"]:
        answer_text = gen_res["text"] or "The model returned an empty response."

        citations = []
        for p in passages:
            citations.append({
                "paper": p["title"],
                "title": p["title"],
                "file": p["file_name"],
                "file_name": p["file_name"],
                "file_type": p.get("file_type", "doc"),
                "page": p["page_num"],
                "page_num": p["page_num"],
                "section": p.get("section", "Context"),
                "excerpt": p["text"][:250],
                "similarity_score": p["similarity_score"]
            })

        insufficient = "not contain sufficient evidence" in answer_text.lower()
        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_passages": passages,
            "insufficient_evidence": insufficient
        }
    else:
        clean_err = sanitize_message(gen_res.get("error", "Gemini generation failed"), key)
        return {
            "answer": f"**Gemini generation failed:** {clean_err}\n\nTop retrieved document passages are available below.",
            "citations": [],
            "retrieved_passages": passages,
            "insufficient_evidence": False
        }


def answer_document_chat(
    query: str,
    vector_store: AcademicVectorStore,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Any = _ENV_DEFAULT,
    model_name: Optional[str] = None,
    top_k: int = 5,
    paper_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None,
    min_similarity: float = 0.0
) -> Dict[str, Any]:
    """
    Multi-Turn Conversational Document Chat:
    Retrieves semantic passages from uploaded multi-format documents, incorporates prior
    chat history turns to handle conversational follow-ups, strictly grounds assertions
    with source coordinates, and explicitly signals when evidence is insufficient without hallucinating.
    """
    passages = vector_store.search_documents(
        query=query,
        top_k=top_k,
        paper_filter=paper_filter,
        file_type_filter=file_type_filter,
        min_similarity=min_similarity,
        api_key=api_key
    )

    if not passages:
        return {
            "answer": "The ingested documents do not contain sufficient evidence to address this question.",
            "citations": [],
            "retrieved_passages": [],
            "insufficient_evidence": True
        }

    # Format passages as numbered evidence blocks
    passage_blocks = []
    for i, p in enumerate(passages, start=1):
        ftype_label = p.get("file_type", "doc").upper()
        passage_blocks.append(
            f"[Source {i}] Document: \"{p['title']}\" | Format: {ftype_label} | Page/Slide/Sheet: {p['page_num']} | File: {p['file_name']} | Section: {p.get('section', 'Context')} | Match Score: {p.get('similarity_score', 1.0):.1%}\n"
            f"Passage Excerpt: \"{p['text']}\"\n"
        )
    context_text = "\n".join(passage_blocks)

    # Format chat history context if present
    history_blocks = []
    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            if content:
                history_blocks.append(f"{role}: {content}")
    history_text = "\n".join(history_blocks) if history_blocks else "None (Initial Inquiry)"

    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    if not key:
        # Fallback offline synthesis
        fallback_lines = [
            "**Notice:** Google Gemini API key is not configured. Displaying top retrieved passages directly from ChromaDB & BM25:\n"
        ]
        citations = []
        for i, p in enumerate(passages, 1):
            ft = p.get("file_type", "doc").upper()
            fallback_lines.append(f"**[{i}] [{ft}] {p['title']} (Unit/Page {p['page_num']})** — Match: `{p['similarity_score']:.1%}` | Section: `{p.get('section', 'General')}`")
            fallback_lines.append(f"> {p['text'][:320]}...\n")
            citations.append({
                "paper": p["title"],
                "title": p["title"],
                "file": p["file_name"],
                "file_name": p["file_name"],
                "file_type": p.get("file_type", "doc"),
                "page": p["page_num"],
                "page_num": p["page_num"],
                "section": p.get("section", "Context"),
                "excerpt": p["text"][:220],
                "similarity_score": p["similarity_score"]
            })

        return {
            "answer": "\n".join(fallback_lines),
            "citations": citations,
            "retrieved_passages": passages,
            "insufficient_evidence": False
        }

    system_prompt = """
You are CausalLens AI's Document Intelligence and RAG Research Assistant.
Answer the user's research inquiry or follow-up question ONLY using the provided verified document excerpts below.

CRITICAL EVIDENCE & CONVERSATIONAL GROUNDING RULES:
1. Ground every claim, parameter, and metric in the provided excerpts.
2. Explicitly cite the source document, format, and unit/page number using:
   [Source N: "<Document Title>", Type: <FORMAT>, Page/Unit: <PageNum>]
3. If the provided excerpts do not contain sufficient evidence to answer the user's question, you MUST clearly state:
   "The ingested documents do not contain sufficient evidence to address this question."
   Do NOT attempt to guess, extrapolate without evidence, or fabricate external facts, citations, or numbers.
4. Support follow-up questions naturally by referencing prior conversation context while remaining strictly faithful to the verified excerpts.
"""

    prompt = f"""
{system_prompt}

PRIOR CONVERSATION HISTORY:
{history_text}

VERIFIED DOCUMENT EXCERPTS:
{context_text}

CURRENT USER QUESTION:
{query}
"""

    gen_res = generate_gemini_text(
        prompt=prompt,
        system_instruction=system_prompt,
        preferred_model=model_name,
        api_key=key,
    )

    if gen_res["success"]:
        answer_text = gen_res["text"] or "The model returned an empty response."

        citations = []
        for p in passages:
            citations.append({
                "paper": p["title"],
                "title": p["title"],
                "file": p["file_name"],
                "file_name": p["file_name"],
                "file_type": p.get("file_type", "doc"),
                "page": p["page_num"],
                "page_num": p["page_num"],
                "section": p.get("section", "Context"),
                "excerpt": p["text"][:250],
                "similarity_score": p["similarity_score"]
            })

        insufficient = "not contain sufficient evidence" in answer_text.lower()
        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_passages": passages,
            "insufficient_evidence": insufficient
        }
    else:
        clean_err = sanitize_message(gen_res.get("error", "Gemini generation failed"), key)
        return {
            "answer": f"**Gemini generation failed:** {clean_err}\n\nTop retrieved document passages are available below.",
            "citations": [],
            "retrieved_passages": passages,
            "insufficient_evidence": False
        }


def extract_key_information(
    document_title_or_id: str,
    vector_store: AcademicVectorStore,
    api_key: Any = _ENV_DEFAULT,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured research metadata and empirical causal parameters from an indexed document:
    - Research Objective & Central Hypotheses
    - Target Population & Sample Size (N)
    - Treatment / Intervention / Exposure (T)
    - Primary & Secondary Outcome Variables (Y)
    - Key Confounders & Adjustment Controls (X)
    - Causal Estimators & Identification Strategy
    - Empirical Findings & Quantitative Estimates
    - Threats to Validity & Disclosed Caveats
    """
    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    # Query for representative passages across the document
    passages = vector_store.search_documents(
        query="objective hypothesis sample size treatment outcome confounder estimator effect size findings limitations",
        top_k=8,
        paper_filter=document_title_or_id if document_title_or_id not in ["All Documents", "All Indexed Papers"] else None,
        api_key=key
    )

    if not passages:
        preview = vector_store.get_document_preview(document_title_or_id, limit_chunks=8)
        if preview and preview.get("chunks_preview"):
            passages = [
                {
                    "title": preview["title"],
                    "file_name": preview["file_name"],
                    "file_type": preview["file_type"],
                    "page_num": c["page_num"],
                    "section": c["section"],
                    "text": c["text"],
                    "similarity_score": 1.0
                }
                for c in preview["chunks_preview"]
            ]

    if not passages:
        return {
            "markdown": f"No indexed passages found for document '{document_title_or_id}'.",
            "extracted_fields": {},
            "passages": []
        }

    excerpts_text = "\n\n".join([
        f"[{p.get('file_type', 'doc').upper()} Unit {p['page_num']}] Section: {p.get('section', 'Context')}\nExcerpt: {p['text']}"
        for p in passages
    ])

    if not key:
        offline_summary = (
            f"### Structured Empirical Parameters: {document_title_or_id}\n\n"
            f"*Ingestion Units:* {len(passages)} passages evaluated.\n\n"
            "| Parameter | Extracted Detail / Document Evidence |\n"
            "| :--- | :--- |\n"
            f"| **Target Document** | `{document_title_or_id}` |\n"
            f"| **Format & Units** | `{passages[0].get('file_type', 'doc').upper()}` (Page/Unit {passages[0].get('page_num', 1)}) |\n"
            f"| **Primary Section** | `{passages[0].get('section', 'General')}` |\n"
            f"| **Sample Excerpt** | {passages[0]['text'][:280]}... |\n\n"
            "> *Note:* Connect your Google Gemini API key to run deep multi-field automated extraction with full parameter parsing.\n"
        )
        return {
            "markdown": offline_summary,
            "extracted_fields": {
                "Document": document_title_or_id,
                "Format": passages[0].get("file_type", "doc").upper()
            },
            "passages": passages
        }

    prompt = f"""
You are a senior econometrician and empirical AI research scientist.
Analyze the following document excerpts and extract structured research metadata and causal parameters.

TARGET DOCUMENT: "{document_title_or_id}"

DOCUMENT EXCERPTS:
{excerpts_text}

INSTRUCTIONS:
Extract the following 8 key dimensions. If a field is not mentioned in the excerpts, write "Not specified in ingested passages".
Do NOT invent details.

1. **Research Objective & Central Hypothesis:** Concise summary of the study or dataset's aim.
2. **Target Population & Sample Size (N):** Cohort definition, sample size, or number of observations.
3. **Treatment / Exposure / Intervention (T):** Definition of the primary treatment or policy variable.
4. **Outcome Variable(s) (Y):** Primary and secondary dependent measures.
5. **Key Confounders & Adjustment Controls (X):** Identified baseline covariates, confounders, or demographic controls.
6. **Causal Estimators & Identification Strategy:** Specific methods mentioned (e.g., Double ML, AIPW, Propensity Score Matching, IPW, OLS, Fixed Effects, Instrumental Variables).
7. **Empirical Findings & Point Estimates:** Quantitative effect sizes, treatment effects, confidence intervals, or p-values.
8. **Threats to Validity & Disclosed Caveats:** Potential unmeasured confounding, selection bias, or data limitations acknowledged.

Format your response with clean Markdown headings, followed by a summary Markdown Table summarizing (Field, Detail, Source Unit).
"""

    gen_res = generate_gemini_text(
        prompt=prompt,
        preferred_model=model_name,
        api_key=key,
    )
    if gen_res["success"]:
        return {
            "markdown": gen_res["text"] or "Extraction returned empty content.",
            "extracted_fields": {},
            "passages": passages
        }
    else:
        clean_err = sanitize_message(gen_res.get("error", "Key information extraction failed"), key)
        return {
            "markdown": f"Key information extraction failed: {clean_err}",
            "extracted_fields": {},
            "passages": passages
        }


def answer_literature_query(
    query: str,
    vector_store: AcademicVectorStore,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
    top_k: int = 5
) -> Dict[str, Any]:
    """Backward-compatible wrapper for answer_document_query."""
    return answer_document_query(
        query=query,
        vector_store=vector_store,
        api_key=api_key,
        model_name=model_name,
        top_k=top_k
    )


# ==============================================================================
# CROSS-DOCUMENT SYNTHESIS & EXECUTIVE REVIEWS
# ==============================================================================
def compare_literature_methods(
    vector_store: AcademicVectorStore,
    comparison_topic: str,
    api_key: Any = _ENV_DEFAULT,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesize a structured cross-document comparison matrix comparing methodologies,
    assumptions, metrics, and empirical findings across uploaded documents.
    """
    docs = vector_store.list_indexed_documents()
    if len(docs) < 1:
        return {
            "comparison_matrix": "Please upload at least one document to generate a cross-document comparison.",
            "papers_evaluated": []
        }

    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    passages = vector_store.search_documents(query=comparison_topic, top_k=10, api_key=key)
    evidence_text = "\n\n".join([f"Document: '{p['title']}' ({p.get('file_type', 'doc').upper()}, Page {p['page_num']}): {p['text']}" for p in passages])

    if not key:
        table_rows = []
        for d in docs:
            table_rows.append(f"| {d['title']} | {d.get('file_type', 'doc').upper()} | Observational Study / Data Asset | Pages/Units: {d['total_pages']}, Chunks: {d['total_chunks']} |")

        fallback_md = (
            "**Notice:** Connect your Gemini API Key in the sidebar for full automated multi-document synthesis.\n\n"
            "### Indexed Document Catalog\n\n"
            "| Document Title | Format | Study / Asset Type | Ingestion Stats |\n"
            "| :--- | :--- | :--- | :--- |\n"
            + "\n".join(table_rows) + "\n\n"
            "**Top Excerpts on Topic:**\n\n"
            + "\n".join([f"- **{p['title']} ({p.get('file_type', 'doc').upper()}, Page {p['page_num']})**: {p['text'][:220]}..." for p in passages[:4]])
        )
        return {
            "comparison_matrix": fallback_md,
            "papers_evaluated": [d["title"] for d in docs]
        }

    prompt = f"""
You are a research synthesis and evidence comparison specialist.
Compare the uploaded documents on the topic: "{comparison_topic}".

EVIDENCE FROM UPLOADED DOCUMENTS:
{evidence_text}

INSTRUCTIONS:
1. Construct a comprehensive Markdown comparison table with columns:
   | Document Title | Format | Causal Estimators / Key Metrics | Core Assumptions / Method | Main Findings & Implications |
2. Follow the table with an analytical synthesis highlighting:
   - Methodological agreement vs. divergence across documents
   - Confounding adjustments, bias mitigation, or data limitations disclosed
   - Relevance to causal inference and decision-making in CausalLens AI.
3. Cite only the provided passages. Do NOT invent documents or citations.
"""

    gen_res = generate_gemini_text(
        prompt=prompt,
        preferred_model=model_name,
        api_key=key,
    )
    if gen_res["success"]:
        return {
            "comparison_matrix": gen_res["text"],
            "papers_evaluated": [d["title"] for d in docs]
        }
    else:
        clean_err = sanitize_message(gen_res.get("error", "Error generating comparison"), key)
        return {
            "comparison_matrix": f"Error generating comparison: {clean_err}",
            "papers_evaluated": [d["title"] for d in docs]
        }


def generate_research_summary(
    paper_title: str,
    vector_store: AcademicVectorStore,
    api_key: Any = _ENV_DEFAULT,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a structured, publication-grade executive review of an indexed document.
    """
    if api_key is _ENV_DEFAULT:
        key = get_api_key()
    else:
        key = api_key

    passages = vector_store.search_documents(
        query="objective methodology results conclusions estimators findings data analysis",
        top_k=8,
        paper_filter=paper_title if paper_title != "All Documents" and paper_title != "All Papers" else None,
        api_key=key
    )

    if not passages:
        return {"summary": f"No indexed passages found for document: {paper_title}."}

    excerpts = "\n\n".join([f"Page/Unit {p['page_num']} ({p.get('file_type', 'doc').upper()}): {p['text']}" for p in passages])

    if not key:
        return {
            "summary": (
                f"### Document Executive Summary: {paper_title}\n\n"
                f"Indexed passages retrieved: {len(passages)}\n\n"
                + "\n\n".join([f"- **Page/Unit {p['page_num']}:** {p['text'][:250]}..." for p in passages[:4]])
            )
        }

    prompt = f"""
You are CausalLens AI's Document Intelligence Reviewer.
Produce a structured, rigorous executive review of the document "{paper_title}" based ONLY on the provided excerpts.

EXCERPTS:
{excerpts}

STRUCTURE YOUR REVIEW WITH:
1. Executive Research / Business Objective
2. Key Methodology, Data Architecture & Assumptions
3. Empirical Results, Quantified Findings & Parameters
4. Methodological Limitations & Confounding Caveats Disclosed
5. Strategic Actionable Takeaways for Decision-Makers
"""

    gen_res = generate_gemini_text(
        prompt=prompt,
        preferred_model=model_name,
        api_key=key,
    )
    if gen_res["success"]:
        return {"summary": gen_res["text"]}
    else:
        clean_err = sanitize_message(gen_res.get("error", "Summary generation failed"), key)
        return {"summary": f"Summary generation failed: {clean_err}"}
