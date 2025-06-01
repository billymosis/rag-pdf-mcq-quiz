import os
import json
import re
import glob
from dotenv import load_dotenv

from llama_cloud_services.parse import ResultType
from rag_backend import config
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
)

from llama_cloud_services import LlamaParse

load_dotenv()
parser = LlamaParse(
    result_type=ResultType.MD,
    api_key=os.getenv("LLAMA_CLOUD_PARSER_KEY", ""),
    verbose=True,
)


def clean_markdown_document(markdown_content):
    """
    Removes repetitive headers, footers (page numbers, organization name),
    and horizontal rules from the markdown document.
    """
    cleaned_content = markdown_content

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

    # Remove standalone horizontal rules (---) that might remain after footer removal,
    # or any other instances that are not part of a header.
    # Be careful not to remove horizontal rules used as part of tables if any exist.
    # Here, we assume "---" is only used for page breaks.
    horizontal_rule_pattern = re.compile(r"^\s*---\s*$", re.MULTILINE)
    cleaned_content = horizontal_rule_pattern.sub("", cleaned_content)

    # Remove any excessive blank lines that might result from removals
    cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)

    return cleaned_content


def write_md_and_json(pdf_path: str, md_path: str, json_path: str):
    # Call the parser to get results for the entire PDF
    results = parser.parse(pdf_path)

    # Initialize an empty list to hold Markdown content from all pages
    all_pages_md_content = []

    pages_json_data = []  # List to hold dictionary representations of each page

    for idx, page in enumerate(results.pages):
        # Assuming page is a Pydantic model or similar object
        # .model_dump() returns a Python dictionary
        pages_json_data.append(page.model_dump())

        # You already have this for markdown, keep it
        all_pages_md_content.append(page.md)
        print(f"  - Processed page {idx + 1}")

    # Create the final Python dictionary structure for JSON
    output_json_dict = {"pages": pages_json_data}

    # Join all the collected Markdown content with a separator (e.g., a newline or page break)
    # You might want to add a clear page separator like '--- Page X ---'
    # For simplicity, we'll just join with two newlines here.
    combined_md = "\n---\n".join(all_pages_md_content)

    # Write the combined Markdown content to the file once
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(combined_md)
    print(f"Successfully wrote combined Markdown to {md_path}")
    # --- WRITE JSON USING json.dump() ---
    with open(json_path, "w", encoding="utf-8") as file:
        # json.dump() writes the Python object directly to the file
        # indent=4 makes the JSON human-readable with 4-space indentation
        json.dump(output_json_dict, file, indent=4)
    print(f"Successfully wrote JSON to {json_path}")


def get_chapter_objectives(text):
    # Find the "Chapter Objectives" section
    match = re.search(
        r"# Chapter Objectives\s*On completion of this chapter you should have a basic knowledge on:\s*",
        text,
    )
    if not match:
        return []

    start_index = match.end()

    # Find the end of the objectives list (before the next heading or "---")
    # This regex looks for a new section starting with '#' or the '---' separator
    end_match = re.search(r"\n(#.+|\n---)", text[start_index:])
    if end_match:
        end_index = start_index + end_match.start()
    else:
        end_index = len(text)  # If no further heading, take till end

    objectives_section = text[start_index:end_index].strip()

    # Extract bullet points
    objectives = re.findall(r"^- (.+)", objectives_section, re.MULTILINE)
    return objectives


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

    documents = set()
    for file_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        documents.add(base_name)

    document_chunks = []

    for base_name in documents:
        pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")
        json_path = os.path.join(pdf_dir, f"{base_name}.json")
        md_path = os.path.join(pdf_dir, f"{base_name}.md")
        if not os.path.exists(md_path):
            try:
                write_md_and_json(
                    pdf_path,
                    md_path,
                    json_path,
                )

            except Exception as e:
                print(f"Error parsing {pdf_path}: {e}")
        else:
            with open(md_path, "r") as file:
                content = file.read()
                objectives = get_chapter_objectives(content)
                cleaned_markdown = clean_markdown_document(content)
                md_header_splits = markdown_splitter.split_text(cleaned_markdown)
                for b in md_header_splits:
                    b.metadata["chapter"] = base_name
                    b.metadata["objectives"] = ", ".join(objectives)
                document_chunks.extend(md_header_splits)
            print(f"Markdown file already exists for {base_name}, skipping parsing.")

    print(f"Split into {len(document_chunks)} chunks.")
    return document_chunks


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
