import os
import time
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


def run_validation():
    """
    Runs the full validation process for the RAG system against all quiz questions.
    Calculates and prints accuracy, and details incorrect answers.
    """
    print("\n--- Starting RAG System Validation ---")

    # Record the start time
    start_time = time.time()

    # Step 1: Ensure API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print(
            "Error: GEMINI_API_KEY environment variable not set. Please set it in your .env file."
        )
        exit(1)

    # Step 2: Load the Vector Database
    # We assume the ChromaDB has already been built by running main.py at least once.
    # If it doesn't exist, this function will raise an error, or you could add logic
    # to build it here if preferred (though it's better to keep validation separate).
    try:
        vector_db = create_or_load_vector_store(
            chunks=[]
        )  # Pass None for chunks as we're loading
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

    correct_predictions = 0
    total_questions = len(quiz_questions)
    incorrect_answers_details = []

    print(f"\n--- Processing {total_questions} quiz questions ---")

    for i, q_data in enumerate(quiz_questions):
        # Use .get() for robustness in case 'expected_answer' or 'answer' is missing
        question_text = q_data.get("question")
        expected_answer = q_data.get("answer")

        if not question_text or not expected_answer:
            print(f"Skipping malformed question at index {i}: {q_data}")
            continue

        print(
            f"\nValidating Question {i+1}/{total_questions} (Chapter: {q_data.get('chapter', 'N/A')})..."
        )

        predicted_answer = answer_quiz_question(question_text, rag_chain)

        if predicted_answer == expected_answer:
            correct_predictions += 1
            print(
                f"  Result: CORRECT (Expected: {expected_answer}, Predicted: {predicted_answer})"
            )
        else:
            incorrect_answers_details.append(
                {
                    "question_num": i + 1,
                    "chapter": q_data.get("chapter", "N/A"),
                    "question": question_text,
                    "expected": expected_answer,
                    "predicted": predicted_answer,
                }
            )
            print(
                f"  Result: INCORRECT (Expected: {expected_answer}, Predicted: {predicted_answer})"
            )
    # Record the end time
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
    print(f"Total Duration: {total_duration:.2f} seconds")  # Print the duration
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
