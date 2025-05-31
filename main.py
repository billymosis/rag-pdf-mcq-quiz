import os
from dotenv import load_dotenv

# Import functions/constants from your new backend modules
from rag_backend import config
from rag_backend.data_processing import (
    load_and_chunk_pdfs,
    load_quiz_questions,
    load_chapter_quiz_questions,
)
from rag_backend.vector_store_manager import (
    create_or_load_vector_store,
    initialize_gemini_llm,
)
from rag_backend.rag_chain import build_rag_chain, answer_quiz_question
from rag_backend.retriever import get_retriever

# Load environment variables from .env file
load_dotenv()


def run_rag():
    """
    Orchestrates the RAG process for testing purposes.
    """
    # Step 1: Load and Chunk PDFs (if ChromaDB doesn't exist)
    # We only load and chunk PDFs if the ChromaDB doesn't already exist.
    # This prevents redundant processing on subsequent runs.
    all_text_chunks = []
    if (
        not os.path.exists(config.PERSIST_DIRECTORY)
        or not os.path.isdir(config.PERSIST_DIRECTORY)
        or not os.listdir(config.PERSIST_DIRECTORY)
    ):
        print("ChromaDB not found or empty. Building from scratch...")
        all_text_chunks = load_and_chunk_pdfs()
    else:
        print("ChromaDB found. Skipping PDF loading and chunking.")

    # Step 2: Create/Load Embeddings and Store in ChromaDB
    vector_db = create_or_load_vector_store(chunks=all_text_chunks)

    # Step 3: Initialize the Gemini LLM
    llm = initialize_gemini_llm()

    retriever = get_retriever(vector_db)

    # Step 4: Build the RAG Chain
    rag_chain = build_rag_chain(llm, retriever)

    # Step 5: Load Quiz Questions for testing
    # You can choose to load all questions or just a specific chapter
    # For initial testing, let's load Chapter 1 questions as before.
    # You can change 'chapter_1' to another chapter key or use load_quiz_questions() for all.
    try:
        quiz_questions_for_test = load_chapter_quiz_questions(chapter_key="chapter_1")
    except KeyError:
        print("Chapter 'chapter_1' not found. Loading all questions for test.")
        quiz_questions_for_test = load_quiz_questions()
        # Take first 10 questions if loading all, to match original test scope
        quiz_questions_for_test = quiz_questions_for_test[:10]

    print("\n--- Testing RAG system with sample questions ---")

    predictions = []
    for q_data in quiz_questions_for_test[0:1]:
        question_text = q_data["question"]
        # Ensure 'expected_answer' key is used, as per the quiz data structure
        expected_answer = q_data["answer"]  # Use 'answer' as fallback

        predicted_answer = answer_quiz_question(question_text, rag_chain)
        predictions.append(
            {
                "question": question_text,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
            }
        )
        print(f"  Expected: {expected_answer}, Predicted: {predicted_answer}\n")

    print("\n--- Sample RAG Test Results ---")
    for p in predictions:
        print(f"Q: {p['question'][:70]}...")
        print(f"  Expected: {p['expected_answer']}, Predicted: {p['predicted_answer']}")
        print("---")

    print(
        "\nRefactoring complete! Your RAG core is now modular."
        "\nNext, you should create and run the validator_script.py to test all questions and analyze accuracy."
    )


if __name__ == "__main__":
    run_rag()
