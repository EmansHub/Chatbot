"""See what an Arabic reranker does to the search results (Arabic chunks only).

For each question: the 5 closest chunks are found by embedding distance (no cutoff), then the
reranker reads the question together with each chunk and gives it a score from 0 to 1.

Run it from the project folder:
    pip install sentence-transformers        # once. It brings PyTorch, so it is a big install
    python test_rerank.py                    # type your own questions
    python test_rerank.py --batch            # run the built-in list and look for a score gap
The first run downloads the model (about 2 GB).

Note: the search question is cleaned (text_normalize.py) like before, but the reranker gets the
ORIGINAL question and chunk text, because it was trained on normally spelled Arabic.
"""
import math
import os
import sys

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from text_normalize import normalize_for_embedding

load_dotenv()

RERANKER = "Omartificial-Intelligence-Space/ARA-Reranker-V1"
EMBED_MODEL = "text-embedding-3-small"   # must be the same model used in ingest_kb.py
LANGUAGE = "ar"
TOP_K = 5

# "yes" = the Arabic document answers it, "no" = it does not. This is my judgment: edit it freely.
QUESTIONS = [
    ("yes", "ما هو الصندوق الاستثماري؟"),
    ("yes", "ما هي الصكوك؟"),
    ("yes", "ما هي صناديق المؤشرات المتداولة؟"),
    ("yes", "ما هي صناديق الاستثمار العقارية المتداولة؟"),
    ("yes", "ما هي أسهم النمو؟"),
    ("yes", "ما هي أسهم العوائد؟"),
    ("yes", "ما هو سوق نمو؟"),
    ("yes", "لماذا يستثمر الناس؟"),
    ("yes", "ما هي مخاطر الاستثمار؟"),
    ("yes", "ماذا تتضمن الخطة المالية؟"),
    ("yes", "ما العلاقة بين العائد والمخاطر؟"),
    ("yes", "من يستطيع الاستثمار في سوق نمو؟"),
    ("yes", "ما هي الأدوات المالية المتاحة في السوق المالية السعودية؟"),
    ("yes", "هل يوجد استثمار بدون مخاطر؟"),
    ("yes", "كيف أوازن بين المخاطر والأرباح؟"),
    ("yes", "ما هي الخطه الماليه؟"),                 # spelling variant
    ("yes", "ماهي الاسهم؟"),                         # spelling variant
    ("yes", "ما هو الصندوق الإستثماري؟"),            # spelling variant
    ("yes", "ما هي الصُّكُوك؟"),                      # diacritics
    ("yes", "وش يعني صناديق المؤشرات؟"),              # Gulf phrasing
    ("no", "ما هو التذبذب؟"),
    ("no", "ما هو تجزئة الأسهم؟"),
    ("no", "ما هو التضخم؟"),
    ("no", "كيف أقرأ القوائم المالية؟"),
    ("no", "ما الفرق بين الصندوق الاستثماري وصناديق المؤشرات المتداولة؟"),
    ("no", "ما هو العائد التاريخي؟"),
    ("no", "كيف أشتري الأسهم؟"),
    ("no", "ما هي رسوم الصناديق الاستثمارية؟"),
    ("no", "ما هو الاكتتاب الأولي؟"),
    ("no", "وين أفضل مطعم بيتزا في الرياض؟"),
]

SEARCH_SQL = """
SELECT heading, content, embedding <=> %s::vector AS distance
FROM knowledge_base.chunks
WHERE language = %s
ORDER BY distance
LIMIT %s
"""


def search(question, embedder):
    """The closest chunks by embedding distance, with no cutoff."""
    vector = embedder.embed_query(normalize_for_embedding(question))
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(SEARCH_SQL, (str(vector), LANGUAGE, TOP_K))
            return cursor.fetchall()
    finally:
        connection.close()


def rerank(model, question, chunks):
    """A relevance score from 0 to 1 for each chunk."""
    scores = [float(s) for s in model.predict([(question, c) for c in chunks])]
    if min(scores) < 0 or max(scores) > 1:            # raw scores: turn them into 0-1
        scores = [1 / (1 + math.exp(-s)) for s in scores]
    return scores


def ranked_results(model, embedder, question):
    rows = search(question, embedder)                  # (heading, content, distance)
    scores = rerank(model, question, [r[1] for r in rows])
    return sorted(zip(scores, rows), key=lambda x: -x[0])


def best_threshold(results):
    """The single score cutoff that makes the fewest mistakes. results: list of (label, top score, question)."""
    thresholds = [i / 100 for i in range(1, 100)]
    errors = {t: [q for label, s, q in results if (s >= t) != (label == "yes")] for t in thresholds}
    fewest = min(len(e) for e in errors.values())
    best = [t for t in thresholds if len(errors[t]) == fewest]
    t = best[len(best) // 2]
    return t, errors[t]


def run_batch(model, embedder):
    results = []
    print(f"{'expect':<7}{'top':>6}{'2nd':>7}  {'dist':>6}  top chunk | question\n")
    for label, question in QUESTIONS:
        ranked = ranked_results(model, embedder, question)
        top_score, (heading, _, distance) = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        results.append((label, top_score, question))
        print(f"{label:<7}{top_score:>6.3f}{second:>7.3f}  {distance:>6.3f}  {heading[:28]} | {question}")

    yes = [(s, q) for label, s, q in results if label == "yes"]
    no = [(s, q) for label, s, q in results if label == "no"]
    low_yes, high_no = min(yes), max(no)
    print(f"\nLowest top score among answerable questions:    {low_yes[0]:.3f}  ({low_yes[1]})")
    print(f"Highest top score among unanswerable questions: {high_no[0]:.3f}  ({high_no[1]})")
    if low_yes[0] > high_no[0]:
        print(f"-> A gap exists. Any cutoff between {high_no[0]:.3f} and {low_yes[0]:.3f} separates them.")
    else:
        print("-> The scores overlap, so no single cutoff separates them.")
    t, wrong = best_threshold(results)
    print(f"Best single cutoff: {t:.2f} -> {len(wrong)} mistake(s) out of {len(results)}")
    for q in wrong:
        print(f"   wrong: {q}")


def run_interactive(model, embedder):
    print("Searching the 'ar' chunks, then reranking the top", TOP_K)
    while True:
        question = input("\nQuestion (press Enter to quit): ").strip()
        if not question:
            break
        for score, (heading, content, distance) in ranked_results(model, embedder, question):
            preview = content.replace("\n", " ")[:80]
            print(f"  {score:.3f}  (dist {distance:.3f})  {heading}  |  {preview}...")


def main():
    print("Loading the reranker (the first run downloads about 2 GB)...")
    model = CrossEncoder(RERANKER)
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    if "--batch" in sys.argv:
        run_batch(model, embedder)
    else:
        run_interactive(model, embedder)


if __name__ == "__main__":
    main()
