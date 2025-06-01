import os
import json
import re
from pathlib import Path
from rag_backend import config
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def clean_markdown_document(markdown_content):
    """
    Removes repetitive headers, footers (page numbers, organization name),
    and horizontal rules from the markdown document.
    """
    cleaned_content = markdown_content

    # 1. Remove the repeating top-level headers.
    # We only want the first instance of these headers, so we target subsequent occurrences.
    # This regex looks for these headers followed by a newline, but not at the very beginning of the string.
    # redundant_header_pattern = re.compile(
    #     r"(?:\n^# RFP Programme - Module 5\s*\n^# Chapter 1 : The Concepts and Fundamentals of Estate Planning\s*\n)",
    #     re.MULTILINE,
    # )
    redundant_header_pattern = re.compile(
        r"^# Chapter \d+.*: .+?\s*\n",  # Removes standalone chapter lines
        re.MULTILINE,
    )
    cleaned_content = redundant_header_pattern.sub("\n", cleaned_content)
    redundant_header_pattern = re.compile(
        r"^# RFP .*\n",  # Removes standalone chapter lines
        re.MULTILINE,
    )
    cleaned_content = redundant_header_pattern.sub("\n", cleaned_content)
    # cleaned_content = redundant_header_pattern.sub(
    #     "\n", cleaned_content, count=1
    # )  # Remove all but the first instance

    # Handle the very first instance of the repeated headers if it exists in a way that
    # doesn't break the initial document structure (e.g., if there's text before it)
    # For your document, it appears clean at the start, so the above should suffice,
    # but a more robust approach might be to remove them and then re-add the first one if needed.
    # For now, let's assume the first instance is desired.

    # 2. Remove page numbers and "Malaysian Financial Planning Council (MFPC)" footer
    # Pattern: "1-X Malaysian Financial Planning Council (MFPC)" or "Malaysian Financial Planning Council (MFPC) 1-X"
    # and the trailing "---" if it's there
    footer_pattern_1 = re.compile(
        r"^\d+-\d+\s+Malaysian Financial Planning Council \(MFPC\)\s*\n?---?\s*$",
        re.MULTILINE,
    )
    footer_pattern_2 = re.compile(
        r"^Malaysian Financial Planning Council \(MFPC\)\s+\d+-\d+\s*\n?---?\s*$",
        re.MULTILINE,
    )
    footer_pattern_3 = re.compile(
        r"^Malaysian Financial Planning Council \(MFPC\)\s*\n?---?\s*$", re.MULTILINE
    )  # Catches just the name if page number is missing

    footer_pattern_4 = re.compile(
        r"\n^Malaysian Financial Planning Council \(MFPC\)\s*\n$", re.MULTILINE
    )

    cleaned_content = footer_pattern_1.sub("", cleaned_content)
    cleaned_content = footer_pattern_2.sub("", cleaned_content)
    cleaned_content = footer_pattern_3.sub("", cleaned_content)
    cleaned_content = footer_pattern_4.sub("", cleaned_content)

    # 3. Remove standalone horizontal rules (---) that might remain after footer removal,
    # or any other instances that are not part of a header.
    # Be careful not to remove horizontal rules used as part of tables if any exist.
    # Here, we assume "---" is only used for page breaks.
    horizontal_rule_pattern = re.compile(r"^\s*---\s*$", re.MULTILINE)
    cleaned_content = horizontal_rule_pattern.sub("", cleaned_content)

    # 4. Remove any excessive blank lines that might result from removals
    cleaned_content = re.sub(
        r"\n{3,}", "\n\n", cleaned_content
    )  # Replace 3 or more newlines with 2

    return cleaned_content


def load_and_chunk_pdfs(
    pdf_dir: str = config.PDF_DIRECTORY,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
):
    """
    Loads all PDF documents from a specified directory, extracts their text,
    and splits the text into smaller, manageable chunks.
    """
    print(f"Loading PDFs from: {pdf_dir}")

    headers_to_split_on = [
        ("#", "Heading 1"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    documents = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".md"):
                _path = Path(os.path.join(root, file))
                with open(_path, "r") as file:
                    content = file.read()
                    cleaned_markdown = clean_markdown_document(content)
                    md_header_splits = markdown_splitter.split_text(cleaned_markdown)
                    documents.extend(md_header_splits)
                    pass

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Try to split by paragraphs, then lines, then words
    )
    final_chunks = []
    for doc in documents:
        chunks_from_header = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata] * len(text_splitter.split_text(doc.page_content)),
        )
        final_chunks.extend(chunks_from_header)
        pass

    print(f"Split into {len(final_chunks)} chunks.")
    return documents


def load_quiz_questions(quiz_path: str = config.QUIZ_QUESTIONS_PATH):
    """
    Loads quiz questions from a JSON file.
    Assumes the JSON is a dictionary where keys are chapter_keys
    and values are lists of question dictionaries. Flattens into a single list.
    """
    if not os.path.exists(quiz_path):
        raise FileNotFoundError(f"Quiz questions file not found at: {quiz_path}")

    with open(quiz_path, "r", encoding="utf-8") as f:
        raw_quiz_data = json.load(f)

    # Flatten the dictionary into a single list of all questions
    all_questions = []
    for chapter_key, questions_list in raw_quiz_data.items():
        arr = []
        for q in questions_list:
            arr.append({**q, "chapter": chapter_key})
        all_questions.extend(arr)

    print(f"Loaded {len(all_questions)} quiz questions from all chapters.")
    return all_questions


# If you only want questions from a specific chapter for testing, you can add a helper:
def load_chapter_quiz_questions(
    chapter_key: str, quiz_path: str = config.QUIZ_QUESTIONS_PATH
):
    """Loads quiz questions for a specific chapter from a JSON file."""
    if not os.path.exists(quiz_path):
        raise FileNotFoundError(f"Quiz questions file not found at: {quiz_path}")

    with open(quiz_path, "r", encoding="utf-8") as f:
        raw_quiz_data = json.load(f)

    if chapter_key not in raw_quiz_data:
        raise KeyError(f"Chapter key '{chapter_key}' not found in quiz data.")

    questions = raw_quiz_data[chapter_key]
    print(f"Loaded {len(questions)} quiz questions for chapter '{chapter_key}'.")
    return questions
