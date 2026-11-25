"""Load a knowledge-base document into the knowledge_base.chunks table.

The document is cut into chunks of the same size with an overlap, using LangChain's
RecursiveCharacterTextSplitter. For Arabic, the text is cleaned for matching
(see text_normalize.py) before it is embedded; the original text is what gets stored.

Run it from the project folder (the one that contains graph.py):
    python ingest_kb.py --dry-run               # English: only show the chunks (no OpenAI call, no database)
    python ingest_kb.py                         # English: embed the chunks and store them (safe to re-run)
    python ingest_kb.py --lang ar --dry-run     # Arabic: the same, for the Arabic document
    python ingest_kb.py --lang ar
Each language replaces only its own rows, so loading Arabic never touches the English ones.
"""
import argparse
import os
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from text_normalize import normalize_for_embedding

load_dotenv()

BASE = Path(__file__).resolve().parent
SOURCE = "Saudi Exchange"
EMBED_MODEL = "text-embedding-3-small"   # gives 1536 numbers per chunk = vector(1536) in the table

# Sizes are in characters. Rule of thumb: chunk_size a bit bigger than the longest paragraph in
# the document (the dry run prints it), so no definition is cut in half. overlap is about 20% of it.
LANGUAGES = {
    "en": {"path": BASE / "information" / "knowledge_base.docx",    "chunk_size": 700, "overlap": 140},
    "ar": {"path": BASE / "information" / "knowledge_base_ar.docx", "chunk_size": 500, "overlap": 100},
}
SKIP_PREFIXES = ("Source:", "المصدر:", "المصدر :")   # lines that are not knowledge (the source has its own column)
MAX_HEADING_WORDS = 10   # a styled line longer than this is body text that only looks like a heading


# ---------- 1. read the document and cut it into chunks ----------
def outline_level(paragraph):
    """0-8 if the line has an outline level (what a table of contents uses), else None."""
    nodes = paragraph._p.xpath("./w:pPr/w:outlineLvl")        # set on the line itself
    style = paragraph.style
    while not nodes and style is not None:                    # or set by its style
        nodes = style.element.xpath("./w:pPr/w:outlineLvl")
        style = style.base_style
    if nodes:
        level = int(nodes[0].get(qn("w:val")))
        return level if level < 9 else None                   # 9 means normal body text
    return None


def is_heading(paragraph):
    return paragraph.style.name.startswith("Heading") or outline_level(paragraph) is not None


def is_list_item(paragraph):
    pPr = paragraph._p.pPr
    return pPr is not None and pPr.numPr is not None


def is_toc(paragraph):
    return paragraph.style.name.upper().startswith("TOC")     # table of contents lines


def read_document(path):
    """Return the whole document as ONE text, where each heading starts, and the body paragraph lengths."""
    parts, headings, body_lengths, position = [], [], [], 0   # headings: list of (start position, heading text)
    for p in Document(path).paragraphs:
        text = p.text.strip()
        if not text or is_toc(p) or text.startswith(SKIP_PREFIXES):
            continue
        if is_heading(p) and not is_list_item(p) and len(text.split()) <= MAX_HEADING_WORDS:
            headings.append((position, text))
            line = text
        else:
            line = ("- " if is_list_item(p) else "") + text
            body_lengths.append(len(line))
        parts.append(line)
        position += len(line) + 2                              # +2 for the blank line between paragraphs
    return "\n\n".join(parts), headings, body_lengths


def load_chunks(path, chunk_size, overlap):
    all_text, headings, _ = read_document(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,                  # tells us where each chunk starts in the text
    )
    chunks = []
    for d in splitter.create_documents([all_text]):
        start = max(d.metadata.get("start_index", 0), 0)
        heading = "(no heading)"               # the section the chunk starts in (for display only)
        for position, text in headings:
            if position <= start:
                heading = text
        chunks.append({"heading": heading, "content": d.page_content})
    return chunks


# ---------- 2. turn the chunks into embeddings ----------
def embed(chunks):
    from langchain_openai import OpenAIEmbeddings

    # Only the text that goes into the embedding is cleaned. The stored text stays as written.
    texts = [normalize_for_embedding(c["content"]) for c in chunks]
    vectors = OpenAIEmbeddings(model=EMBED_MODEL).embed_documents(texts)
    if len(vectors[0]) != 1536:
        sys.exit(f"The model returned {len(vectors[0])} numbers per chunk, but the table expects 1536.")
    return vectors


# ---------- 3. save them in the database ----------
def store(chunks, vectors, language):
    import psycopg2

    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    try:
        with connection:                  # saves everything if it all works, undoes it if anything fails
            with connection.cursor() as cursor:
                # delete the old copy of THIS language first, so running the script again never creates duplicates
                cursor.execute(
                    "DELETE FROM knowledge_base.chunks WHERE source = %s AND language = %s",
                    (SOURCE, language),
                )
                for chunk, vector in zip(chunks, vectors):
                    cursor.execute(
                        """INSERT INTO knowledge_base.chunks (heading, content, embedding, source, language)
                           VALUES (%s, %s, %s::vector, %s, %s)""",
                        (chunk["heading"], chunk["content"], str(vector), SOURCE, language),
                    )
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANGUAGES, default="en")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = LANGUAGES[args.lang]
    path = config["path"]
    if not path.exists():
        sys.exit(f"File not found: {path}")

    all_text, _, body_lengths = read_document(path)
    chunks = load_chunks(path, config["chunk_size"], config["overlap"])
    sizes = [len(c["content"]) for c in chunks]
    print(f"Language: {args.lang} | file: {path.name}")
    print(f"Longest paragraph: {max(body_lengths)} characters | chunk size {config['chunk_size']}, overlap {config['overlap']}"
          + ("  <-- paragraph is longer than the chunk size: raise chunk_size" if max(body_lengths) > config["chunk_size"] else ""))
    print(f"Found {len(chunks)} chunks (characters per chunk: min {min(sizes)}, max {max(sizes)}):\n")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["content"].replace("\n", " ")[:60]
        print(f"{i:2}. [{chunk['heading']}]  {len(chunk['content'])} chars  |  {preview}...")

    if args.lang == "ar":
        latin = sorted(set(re.findall(r"[A-Za-z][A-Za-z0-9\-]*", all_text)))
        if latin:
            print(f"\nWarning: Latin text found in the Arabic document: {', '.join(latin[:15])}")
            print("Arabic replies must not contain Latin letters, so these may leak into answers.")
        print("\nWhat gets embedded for the first chunk (cleaned):")
        print(normalize_for_embedding(chunks[0]["content"])[:120].replace("\n", " "))

    if args.dry_run:
        print("\nDry run: nothing was sent or saved.")
        return

    print("\nEmbedding...")
    vectors = embed(chunks)
    store(chunks, vectors, args.lang)
    print(f"Saved {len(chunks)} chunks ({args.lang}) to knowledge_base.chunks.")


if __name__ == "__main__":
    main()
