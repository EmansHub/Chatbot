"""Try the search: type a question, see the 3 closest chunks and their distances.

Run it from the project folder:
    python test_search.py        # searches the English chunks
    python test_search.py ar     # searches the Arabic chunks
Smaller distance = closer match. Write the numbers down: you need them to set the cutoff.
"""
import os
import sys

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from text_normalize import normalize_for_embedding

load_dotenv()

LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else "en"
EMBED_MODEL = "text-embedding-3-small"   # must be the same model used in ingest_kb.py
embedder = OpenAIEmbeddings(model=EMBED_MODEL)

SEARCH_SQL = """
SELECT heading, content, embedding <=> %s::vector AS distance
FROM knowledge_base.chunks
WHERE language = %s
ORDER BY distance
LIMIT 3
"""


def search(question):
    vector = embedder.embed_query(normalize_for_embedding(question))   # the question is cleaned like the document
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(SEARCH_SQL, (str(vector), LANGUAGE))
            return cursor.fetchall()
    finally:
        connection.close()


print(f"Searching the '{LANGUAGE}' chunks.")
while True:
    question = input("\nQuestion (press Enter to quit): ").strip()
    if not question:
        break
    for heading, content, distance in search(question):
        preview = content.replace("\n", " ")[:90]
        print(f"  {distance:.3f}  {heading}  |  {preview}...")
