"""
Pattern 03: Excel Spreadsheet RAG
Demonstrates:
1. Ingesting tabular data from Excel (.xlsx) workbooks using openpyxl.
2. Serializing structured rows into textual knowledge context.
3. Querying specific record attributes accurately with zero hallucination.
"""

import os
from pathlib import Path
import sys
import openpyxl
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

BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "cohere_login_data.xlsx"

if not EXCEL_PATH.is_file():
    raise FileNotFoundError(f"Excel file not found at: {EXCEL_PATH}")

def load_excel_records(path: Path) -> list[str]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    records = []

    for row in ws.iter_rows(min_row=2, values_only=True):  # Skip header row
        if len(row) >= 2 and row[0] is not None and row[1] is not None:
            username = str(row[0]).strip()
            password = str(row[1]).strip()
            records.append(f"Username: {username}, Password: {password}")

    wb.close()
    return records


if __name__ == "__main__":
    print(f"Loading spreadsheet from: {EXCEL_PATH}")
    knowledge_base = load_excel_records(EXCEL_PATH)
    print(f"Total User Records Ingested: {len(knowledge_base)}")

    question = "What is the password for Username: happyhours?"
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    # Filter or combine context
    context = "\n".join(knowledge_base)
    prompt = f"""
Answer the question using ONLY the provided spreadsheet records below.
If the account is not found, reply 'Account not found in the records.'

Spreadsheet Records:
{context}

Question:
{question}

Answer:
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text

    print("\n[Cohere Grounded Answer]:")
    print(answer.strip())
