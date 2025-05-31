import os
import json
from pathlib import Path
from rag_backend import config
from langchain_community.document_loaders import UnstructuredMarkdownLoader


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

    documents = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".md"):
                _path = Path(os.path.join(root, file))
                loader = UnstructuredMarkdownLoader(
                    _path, chunks=chunk_size, chunk_overlap=chunk_overlap
                )
                document = loader.load()
                documents.extend(document)

    print(f"Loaded {len(documents)} initial pages/documents.")

    print(f"Split into {len(documents)} chunks.")
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
