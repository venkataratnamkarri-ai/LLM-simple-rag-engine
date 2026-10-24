"""
Pattern 02: PDF Document RAG with Chunking
Demonstrates:
1. Extracting text from multi-page PDF documents using PyPDF2.
2. Sliding-window text chunking with configurable size and overlap.
3. Lexical keyword scoring over document chunks.
4. Answering domain questions strictly grounded in PDF content.
"""

import os
from pathlib import Path
import sys
import PyPDF2
import cohere
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")

co = cohere.ClientV2(api_key=api_key)

# 1. Resolve PDF Path
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Agile_Process_Guide_5_Pages.pdf"

if not PDF_PATH.is_file():
    raise FileNotFoundError(f"PDF document not found at: {PDF_PATH}")

# 2. Extract Text from PDF
def extract_pdf_text(path: Path) -> str:
    pdf_text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
    return pdf_text

# 3. Sliding Window Text Chunker
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# 4. Keyword-based Retrieval
def retrieve_chunks(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    keywords = query.lower().split()
    scored = []
    for chunk in chunks:
        score = sum(1 for word in keywords if word.strip("?,.!") in chunk.lower())
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

# 5. Answer Generation
def generate_pdf_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n--- CHUNK ---\n\n".join(context_chunks)
    prompt = f"""
You are answering questions using ONLY the supplied context from an Agile Process Guide.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, reply exactly:
I couldn't find this information in the document.
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text
    return answer.strip()


if __name__ == "__main__":
    print(f"Reading PDF from: {PDF_PATH}")
    full_text = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(full_text, chunk_size=800, overlap=100)
    print(f"Total Chunks Created: {len(chunks)}")

    question = (
        "Tell the exact role and responsibilities of Scrum Master and "
        "what kind of value Scrum Master brings to the team."
    )

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    top_chunks = retrieve_chunks(question, chunks, top_k=3)
    print(f"\n[Retrieved {len(top_chunks)} Chunks]:")
    for idx, chunk in enumerate(top_chunks, 1):
        preview = chunk[:200].replace("\n", " ")
        print(f"  Chunk {idx}: {preview}...")

    answer = generate_pdf_answer(question, top_chunks)
    print("\n[Grounded LLM Answer]:")
    print(answer)
