"""Compare embedding models on your Arabic chunks. Nothing is written to the database.

The Arabic chunks are read from the database. The chunks and the test questions are then embedded
in memory with each model, and for every question we check:
  - does the chunk that holds the answer appear in the top 3? (this is what the answer step needs)
  - is there a gap between the best distance of answerable and unanswerable questions?

Models compared:
  openai      your current model (chunks use the vectors already stored; the question is cleaned first)
  gate-raw    GATE-AraBERT-v1 (Arabic, free, runs locally) on the original text
  gate-norm   the same model on text cleaned with text_normalize.py
  bge-m3      optional (--bge): a multilingual retrieval model (free, about a 2 GB download)

Run it from the project folder:
    python test_embeddings.py
    python test_embeddings.py --bge
"""
import json
import os
import sys

import numpy as np
import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from text_normalize import normalize_for_embedding

load_dotenv()

TOP = 3
# (expected, question, a phrase from the document that the answer chunk must contain)
QUESTIONS = [
    ("yes", "ما هو الصندوق الاستثماري؟", "هي محفظة استثمارية"),
    ("yes", "ما هي الصكوك؟", "أدوات تمويلية"),
    ("yes", "ما هي صناديق المؤشرات المتداولة؟", "هي صناديق استثمارية تتكون"),
    ("yes", "ما هي صناديق الاستثمار العقارية المتداولة؟", "هي صناديق استثمارية عقارية"),
    ("yes", "ما هي أسهم النمو؟", "أسهم النمو"),
    ("yes", "ما هي أسهم العوائد؟", "أسهم العوائد"),
    ("yes", "ما هو سوق نمو؟", "نمو هي سوق"),
    ("yes", "لماذا يستثمر الناس؟", "قد يساعد الاستثمار الجيد"),
    ("yes", "ما هي مخاطر الاستثمار؟", "خسارة رأس المال"),
    ("yes", "ماذا تتضمن الخطة المالية؟", "خطة مالية مكتوبة"),
    ("yes", "ما العلاقة بين العائد والمخاطر؟", "علاقة مباشرة بين العائد"),
    ("yes", "من يستطيع الاستثمار في سوق نمو؟", "مخصص للمستثمرين المؤهلين"),
    ("yes", "ما هي الأدوات المالية المتاحة في السوق المالية السعودية؟", "الأدوات المتاحة للاستثمار"),
    ("yes", "هل يوجد استثمار بدون مخاطر؟", "لا يوجد استثمار بدون مخاطر"),
    ("yes", "كيف أوازن بين المخاطر والأرباح؟", "تنويع"),
    ("yes", "ما هي الخطه الماليه؟", "خطة مالية مكتوبة"),
    ("yes", "ماهي الاسهم؟", "هي عبارة عن استثمارات تمنح"),
    ("yes", "ما هو الصندوق الإستثماري؟", "هي محفظة استثمارية"),
    ("yes", "ما هي الصُّكُوك؟", "أدوات تمويلية"),
    ("yes", "وش يعني صناديق المؤشرات؟", "هي صناديق استثمارية تتكون"),
    ("no", "ما هو التذبذب؟", None),
    ("no", "ما هو تجزئة الأسهم؟", None),
    ("no", "ما هو التضخم؟", None),
    ("no", "كيف أقرأ القوائم المالية؟", None),
    ("no", "ما الفرق بين الصندوق الاستثماري وصناديق المؤشرات المتداولة؟", None),
    ("no", "ما هو العائد التاريخي؟", None),
    ("no", "كيف أشتري الأسهم؟", None),
    ("no", "ما هي رسوم الصناديق الاستثمارية؟", None),
    ("no", "ما هو الاكتتاب الأولي؟", None),
    ("no", "وين أفضل مطعم بيتزا في الرياض؟", None),
]


def load_chunks():
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT heading, content, embedding::text FROM knowledge_base.chunks "
                           "WHERE language = 'ar' ORDER BY id")
            return cursor.fetchall()
    finally:
        connection.close()


def unit(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def local_model(name, texts_q, texts_c):
    model = SentenceTransformer(name)
    return (np.asarray(model.encode(texts_q, normalize_embeddings=True)),
            np.asarray(model.encode(texts_c, normalize_embeddings=True)))


def evaluate(q_vecs, c_vecs, contents):
    """For every question: the best distance and whether the answer chunk is in the top 3."""
    dist = 1 - unit(q_vecs) @ unit(c_vecs).T              # cosine distance, same idea as pgvector's <=>
    rows = []
    for i, (label, question, key) in enumerate(QUESTIONS):
        order = np.argsort(dist[i])
        top_ids = order[:TOP]
        in_top1 = key is not None and key in contents[order[0]]
        in_top3 = key is not None and any(key in contents[j] for j in top_ids)
        rows.append({"label": label, "question": question, "best": float(dist[i][order[0]]),
                     "top1": in_top1, "top3": in_top3})
    return rows


def best_cutoff(rows):
    thresholds = [round(t, 3) for t in np.arange(0.05, 1.0, 0.005)]
    errors = {t: [r["question"] for r in rows if (r["best"] <= t) != (r["label"] == "yes")] for t in thresholds}
    fewest = min(len(e) for e in errors.values())
    best = [t for t in thresholds if len(errors[t]) == fewest]
    return best[len(best) // 2], fewest


def main():
    chunks = load_chunks()
    if not chunks:
        sys.exit("No Arabic chunks found. Run: python ingest_kb.py --lang ar")
    contents = [c[1] for c in chunks]
    questions = [q for _, q, _ in QUESTIONS]
    print(f"{len(chunks)} Arabic chunks, {len(QUESTIONS)} questions\n")

    results = {}
    openai = OpenAIEmbeddings(model="text-embedding-3-small")
    q_open = openai.embed_documents([normalize_for_embedding(q) for q in questions])
    c_open = [json.loads(c[2]) for c in chunks]           # the vectors already stored in the database
    results["openai"] = evaluate(q_open, c_open, contents)

    print("Loading GATE-AraBERT-v1 (the first run downloads about 540 MB)...")
    gate = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
    results["gate-raw"] = evaluate(*local_model(gate, questions, contents), contents)
    results["gate-norm"] = evaluate(*local_model(gate, [normalize_for_embedding(q) for q in questions],
                                                 [normalize_for_embedding(c) for c in contents]), contents)
    if "--bge" in sys.argv:
        print("Loading bge-m3 (the first run downloads about 2 GB)...")
        results["bge-m3"] = evaluate(*local_model("BAAI/bge-m3", questions, contents), contents)

    names = list(results)
    print("\nBest distance per question (smaller = closer):")
    print(f"{'expect':<7}" + "".join(f"{n:>11}" for n in names) + "  question")
    for i, (label, question, _) in enumerate(QUESTIONS):
        print(f"{label:<7}" + "".join(f"{results[n][i]['best']:>11.3f}" for n in names) + f"  {question}")

    print("\n==== SUMMARY (20 answerable questions: does the answer chunk reach the answer step?) ====")
    for n in names:
        rows = results[n]
        yes = [r for r in rows if r["label"] == "yes"]
        no = [r for r in rows if r["label"] == "no"]
        top1, top3 = sum(r["top1"] for r in yes), sum(r["top3"] for r in yes)
        low_yes, high_no = max(r["best"] for r in yes), min(r["best"] for r in no)
        cut, mistakes = best_cutoff(rows)
        gap = "gap" if low_yes < high_no else "overlap"
        print(f"{n:<10} answer chunk in top 1: {top1}/20 | in top 3: {top3}/20 | "
              f"answerable up to {low_yes:.3f}, unanswerable from {high_no:.3f} ({gap}) | "
              f"best cutoff {cut:.3f} -> {mistakes} mistake(s)")
    for n in names:
        missed = [r["question"] for r in results[n] if r["label"] == "yes" and not r["top3"]]
        if missed:
            print(f"\n{n}: answer chunk NOT in the top 3 for:")
            for q in missed:
                print(f"   {q}")


if __name__ == "__main__":
    main()
