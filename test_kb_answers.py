"""End-to-end test of the knowledge-base answer step for Arabic.

For each question: find the closest Arabic chunks (loose cutoff), give them to the SAME strict
answer prompt and model that graph.py uses, and check whether the bot answered or refused.
This measures what the user would actually see, instead of looking at score gaps.

Run it from the project folder:
    python test_kb_answers.py                       # cutoff 0.8, top 5 chunks
    python test_kb_answers.py --cutoff 0.7 --top 3  # try other settings
It needs graph.py to import without errors (it reuses its model, prompt and embedder).
"""
import argparse
import os

import psycopg2
from langchain_core.messages import HumanMessage, SystemMessage

from graph import KB_ANSWER_PROMPT, KB_NO_INFO, kb_embedder, llm
from text_normalize import normalize_for_embedding

LANGUAGE = "ar"

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


def retrieve(question, top, cutoff):
    """The closest chunks (heading, content, distance) and the ones within the cutoff."""
    vector = kb_embedder.embed_query(normalize_for_embedding(question))
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(SEARCH_SQL, (str(vector), LANGUAGE, top))
            rows = cursor.fetchall()
    finally:
        connection.close()
    return rows, [r for r in rows if r[2] <= cutoff]


def answer(question, kept):
    """What the knowledge-base answer step would reply, using only the kept chunks."""
    if not kept:
        return KB_NO_INFO                                # same shortcut as the real step: no model call
    passages = "\n\n---\n\n".join(r[1] for r in kept)
    return llm.invoke([
        SystemMessage(content=KB_ANSWER_PROMPT.format(passages=passages)),
        HumanMessage(content=question),
    ]).content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=0.8)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    print(f"Arabic chunks | top {args.top} | cutoff {args.cutoff}\n")
    results = []                                          # (label, refused, question, reply, rows, kept)
    for label, question in QUESTIONS:
        rows, kept = retrieve(question, args.top, args.cutoff)
        reply = answer(question, kept)
        refused = not reply or any(m in reply.lower() for m in (KB_NO_INFO.lower(), "لا تتوفر لدي معلومات", "ليس لدي معلومات", "لا أملك معلومات", "ما عندي معلومات"))
        results.append((label, refused, question, reply, rows, kept))
        good = (label == "yes") != refused                # answered a "yes", or refused a "no"
        best = f"{rows[0][2]:.3f}" if rows else "-"
        print(f"{'OK ' if good else 'BAD'} expect={label:<3} {'refused ' if refused else 'answered'} "
              f"chunks={len(kept)}/{len(rows)} best={best} | {question}")
        print(f"      -> {reply[:170].replace(chr(10), ' ')}")

    yes = [r for r in results if r[0] == "yes"]
    no = [r for r in results if r[0] == "no"]
    false_refusals = [r for r in yes if r[1]]
    leaks = [r for r in no if not r[1]]
    print("\n==== SUMMARY ====")
    print(f"Answerable questions answered:   {len(yes) - len(false_refusals)}/{len(yes)}")
    print(f"Unanswerable questions refused:  {len(no) - len(leaks)}/{len(no)}")
    print(f"Correct overall: {len(results) - len(false_refusals) - len(leaks)}/{len(results)}")
    print("(Answered does not mean correct: read the answers above and check they use only the document.)")

    if false_refusals:
        print("\nFalse refusals (the document answers it, the bot refused):")
        for _, _, q, _, rows, kept in false_refusals:
            why = "no chunk was within the cutoff" if not kept else "the answer step refused"
            heads = ", ".join(f"{r[0][:18]} ({r[2]:.2f})" for r in rows[:3])
            print(f"  {q}\n     reason: {why} | closest: {heads}")
    if leaks:
        print("\nLeaks (the document does not answer it, the bot answered):")
        for _, _, q, reply, _, _ in leaks:
            print(f"  {q}\n     -> {reply[:300].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
