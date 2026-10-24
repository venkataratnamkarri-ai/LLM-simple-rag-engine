"""
Pattern 05: REST API RAG - User Directory
Demonstrates:
1. Dynamic retrieval of live REST API data (DummyJSON Users API).
2. Transforming nested JSON responses into clean structured document context.
3. Querying specific entity attributes using Cohere ClientV2.
"""

import os
import sys
import requests
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

API_URL = "https://dummyjson.com/users?limit=30"

def fetch_user_documents() -> list[str]:
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()
    data = response.json()
    users = data.get("users", [])

    documents = []
    for user in users:
        doc = (
            f"Name: {user.get('firstName')} {user.get('lastName')} | "
            f"Age: {user.get('age')} | "
            f"Email: {user.get('email')} | "
            f"Phone: {user.get('phone')} | "
            f"City: {user.get('address', {}).get('city', 'N/A')} | "
            f"Company: {user.get('company', {}).get('name', 'N/A')} | "
            f"EyeColor: {user.get('eyeColor', 'N/A')}"
        )
        documents.append(doc)
    return documents


if __name__ == "__main__":
    print(f"Fetching users from API: {API_URL}")
    documents = fetch_user_documents()
    print(f"Total User Records Retrieved: {len(documents)}")

    question = "Which users have blue eye color? List their full name, company, and city."
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    context = "\n".join(documents)
    prompt = f"""
Context:
{context}

Question:
{question}

Answer ONLY from the supplied context:
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

    print("\n[Cohere Answer]:")
    print(answer.strip())
