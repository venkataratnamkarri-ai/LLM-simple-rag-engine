"""
Pattern 04: CSV Dataset RAG with Pandas
Demonstrates:
1. Ingesting agile user stories from CSV using Pandas.
2. Indexing text columns for lexical document retrieval.
3. Querying requirements and user stories with zero hallucination constraints.
"""

import os
from pathlib import Path
import sys
import pandas as pd
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
CSV_PATH = BASE_DIR / "data" / "db_userstories.csv"

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"User stories CSV not found at: {CSV_PATH}")

def load_user_stories(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "story" not in df.columns:
        raise ValueError("CSV must contain a 'story' column")
    return df["story"].dropna().astype(str).tolist()

def retrieve_stories(query: str, stories: list[str], k: int = 5) -> list[str]:
    keywords = query.lower().split()
    scored_docs = []

    for doc in stories:
        score = sum(1 for word in keywords if word.strip("?,.!") in doc.lower())
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored_docs[:k]]

def generate_rag_answer(question: str, documents: list[str]) -> str:
    context = "\n".join(f"- {doc}" for doc in documents)
    prompt = f"""
Answer ONLY from the context below.

If the answer is not present in the context, reply:
'I couldn't find this information in the supplied context.'

Context:
{context}

Question:
{question}

Answer:
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = ""
    for item in response.message.content:
        if hasattr(item, "text"):
            answer += item.text
    return answer.strip()


if __name__ == "__main__":
    print(f"Loading user stories from: {CSV_PATH}")
    stories = load_user_stories(CSV_PATH)
    print(f"Total Stories Loaded: {len(stories)}")

    question = "List down Login and authentication related User Stories"
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    matched_stories = retrieve_stories(question, stories, k=5)
    print(f"\n[Retrieved {len(matched_stories)} Matched Stories]:")
    for doc in matched_stories:
        print(f"  • {doc}")

    answer = generate_rag_answer(question, matched_stories)
    print("\n[Cohere Answer]:")
    print(answer)
