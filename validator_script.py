import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Import functions and configurations from your RAG backend modules
from rag_backend import config
from rag_backend.data_processing import load_quiz_questions
from rag_backend.vector_store_manager import (
    create_or_load_vector_store,
    initialize_gemini_llm,
)
from rag_backend.rag_chain import build_rag_chain, answer_quiz_question

# Load environment variables from .env file
load_dotenv()


def process_question(q_data, rag_chain, question_num, total_questions):
    """
    Thread-safe function to process a single question
    Returns tuple: (result_dict, is_correct)
    """
    question_text = q_data.get("question")
    expected_answer = q_data.get("answer")

    if not question_text or not expected_answer:
        print(f"Skipping malformed question at index {question_num-1}: {q_data}")
        return None, False

    print(
        f"\nValidating Question {question_num}/{total_questions} (Chapter: {q_data.get('chapter', 'N/A')})..."
    )

    predicted_answer = answer_quiz_question(question_text, rag_chain)
    is_correct = predicted_answer == expected_answer

    result = {
        "question_num": question_num,
        "chapter": q_data.get("chapter", "N/A"),
        "question": question_text,
        "expected": expected_answer,
        "predicted": predicted_answer,
    }

    print(
        f"  Result: {'CORRECT' if is_correct else 'INCORRECT'} (Expected: {expected_answer}, Predicted: {predicted_answer})"
    )

    return result, is_correct


def run_validation():
    """
    Runs the full validation process for the RAG system against all quiz questions.
    Calculates and prints accuracy, and details incorrect answers.
    """
    print("\n--- Starting RAG System Validation ---")
    start_time = time.time()

    # Step 1: Ensure API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print(
            "Error: GEMINI_API_KEY environment variable not set. Please set it in your .env file."
        )
        exit(1)

    # Step 2: Load the Vector Database
    try:
        vector_db = create_or_load_vector_store(chunks=[])
    except ValueError as e:
        print(f"Error loading vector store: {e}")
        print(
            "Please ensure you have run 'python main.py' at least once to build the ChromaDB."
        )
        exit(1)

    # Step 3: Initialize the Gemini LLM
    llm = initialize_gemini_llm()

    # Step 4: Build the RAG Chain
    rag_chain = build_rag_chain(vector_db, llm)

    # Step 5: Load ALL Quiz Questions
    quiz_questions = load_quiz_questions(config.QUIZ_QUESTIONS_PATH)
    total_questions = len(quiz_questions)
    print(f"\n--- Processing {total_questions} quiz questions ---")

    # Threading implementation
    correct_predictions = 0
    incorrect_answers_details = []

    # Determine optimal thread count (adjust based on your system)
    num_threads = min(8, os.cpu_count() or 4)  # Use up to 8 threads

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, q_data in enumerate(quiz_questions):
            futures.append(
                executor.submit(
                    process_question,
                    q_data=q_data,
                    rag_chain=rag_chain,
                    question_num=i + 1,
                    total_questions=total_questions,
                )
            )

        # Collect results as they complete
        for future in futures:
            result, is_correct = future.result()
            if result:  # Skip None results from malformed questions
                if is_correct:
                    correct_predictions += 1
                else:
                    incorrect_answers_details.append(result)

    # Calculate statistics
    end_time = time.time()
    total_duration = end_time - start_time
    accuracy = (
        (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
    )

    print("\n" + "=" * 40)
    print("--- RAG System Validation Summary ---")
    print(f"Total Questions: {total_questions}")
    print(f"Correctly Answered: {correct_predictions}")
    print(f"Incorrectly Answered: {total_questions - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(
        f"Total Duration: {total_duration:.2f} seconds (Threading: {num_threads} threads)"
    )
    print("=" * 40)

    if incorrect_answers_details:
        print("\n--- Details of Incorrect Answers ---")
        for detail in incorrect_answers_details:
            print(
                f"Q{detail['question_num']} (Chapter: {detail['chapter']}): {detail['question'][:120]}..."
            )
            print(f"  Expected: {detail['expected']}, Predicted: {detail['predicted']}")
            print("-" * 20)

    print(
        "\nValidation complete. Use the incorrect answer details to debug and improve your RAG system."
    )


if __name__ == "__main__":
    run_validation()
