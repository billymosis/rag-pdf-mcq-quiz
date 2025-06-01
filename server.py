from contextlib import asynccontextmanager
from operator import is_
import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional

from pydantic.fields import Field

# Import functions and configurations from your rag_backend modules
from rag_backend import config
from rag_backend.data_processing import load_and_chunk_pdfs
from rag_backend.vector_store_manager import (
    create_or_load_vector_store,
    initialize_gemini_llm,
)
from rag_backend.rag_chain import build_rag_chain, answer_quiz_question

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes RAG components when the FastAPI application starts.
    """
    global vector_db, llm, rag_chain
    print("FastAPI app starting up. Initializing RAG components...")

    try:
        # Step 1: Load and Chunk PDFs (if ChromaDB doesn't exist)
        # This part should ideally only run if the vector store needs building.
        # Your `create_or_load_vector_store` handles this check.
        all_text_chunks = []
        if (
            not os.path.exists(config.PERSIST_DIRECTORY)
            or not os.path.isdir(config.PERSIST_DIRECTORY)
            or not os.listdir(config.PERSIST_DIRECTORY)
        ):
            print("ChromaDB not found or empty. Building from scratch...")
            all_text_chunks = load_and_chunk_pdfs()

        vector_db = create_or_load_vector_store(
            chunks=all_text_chunks
        )  # all_text_chunks can be empty if DB exists

        # Step 3: Initialize the Gemini LLM
        llm = initialize_gemini_llm()

        # Step 4: Build the RAG Chain
        rag_chain = build_rag_chain(vector_db, llm)

        print("RAG components initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RAG components: {e}")
        # In a real app, you might want to log this and potentially exit gracefully
        # raise Exception(f"Application startup failed: {e}") # Or handle more gracefully
    yield
    print("FastAPI app shutting down.")

    # Model


class RagBuildRequest(BaseModel):
    """
    Model for the RAG build request.
    Could include options like specifying a new PDF directory if needed.
    """

    pdf_directory: Optional[str] = config.PDF_DIRECTORY
    rebuild_db: bool = False  # Set to True to force a rebuild even if DB exists


class QuizQuestionRequest(BaseModel):
    """
    Model for a quiz question request.
    """

    question: str


class QuizAnswerResponse(BaseModel):
    """
    Model for a quiz answer response.
    """

    question: str
    predicted_option: str
    # You might add more fields like source_documents, raw_llm_response for debugging/transparency
    # raw_llm_response: Optional[str] = None
    # source_documents: Optional[List[Dict]] = None


class RagBuildResponse(BaseModel):
    """
    Model for the RAG build response.
    """

    status: str
    message: str
    chunks_processed: Optional[int] = None


# For batch questions
class BatchQuestionItem(BaseModel):
    """
    Model for an item in the batch question request.
    Includes an optional 'answer' for validation/testing.
    """

    question: str
    answer: Optional[str] = None  # 'answer' is the expected answer for validat


class BatchQuizRequest(BaseModel):
    """
    Model for a batch of quiz questions.
    """

    questions: List[BatchQuestionItem] = Field(..., min_length=1)


class BatchQuizAnswerItem(BaseModel):
    """
    Model for an item in the batch quiz answer response.
    """

    question: str
    expected_answer: Optional[str] = None  # The 'answer' from the input, for comparison
    predicted_option: str
    is_correct: Optional[bool] = (
        None  # True if predicted_option matches expected_answer
    )
    # raw_llm_response: Optional[str] = None # Optional for debugging
    # source_documents: Optional[List[Dict]] = None # Optional for debugging


class BatchQuizAnswerResponse(BaseModel):
    """
    Model for a batch quiz answer response.
    """

    total_questions: int
    correct_predictions: Optional[int] = None  # Only if expected answers are provided
    accuracy: Optional[float] = None  # Only if expected answers are provided
    results: List[BatchQuizAnswerItem]


# --- Global RAG Components (initialized once) ---
# These will be loaded when the FastAPI app starts
vector_db = None
llm = None
rag_chain = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Quiz Assistant API",
    description="Backend API for a RAG-powered quiz assistant for estate planning.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG Quiz Assistant API! Visit /docs for API documentation."
    }


@app.post("/rag/build", response_model=RagBuildResponse, status_code=status.HTTP_200_OK)
async def build_rag_knowledge_base(request: RagBuildRequest):
    """
    Endpoint to build or rebuild the RAG knowledge base from PDF documents.
    This can be a long-running operation.
    """
    global vector_db, rag_chain

    if (
        not request.rebuild_db
        and os.path.exists(config.PERSIST_DIRECTORY)
        and os.path.isdir(config.PERSIST_DIRECTORY)
        and os.listdir(config.PERSIST_DIRECTORY)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="ChromaDB already exists. Set 'rebuild_db' to true in the request body to force a rebuild.",
        )

    try:
        print(f"Building RAG knowledge base from directory: {request.pdf_directory}")
        # This function should be adjusted to return the actual chunks if it doesn't already
        all_text_chunks = load_and_chunk_pdfs(pdf_dir=request.pdf_directory)

        # Re-create/load vector store with the new chunks
        vector_db = create_or_load_vector_store(
            chunks=all_text_chunks,
            persist_directory=config.PERSIST_DIRECTORY,  # Ensure this matches config
        )
        # Rebuild the RAG chain with the new vector_db
        if llm:  # Ensure LLM is initialized
            rag_chain = build_rag_chain(vector_db, llm)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM not initialized. Cannot rebuild RAG chain.",
            )

        return RagBuildResponse(
            status="success",
            message="RAG knowledge base built/rebuilt successfully.",
            chunks_processed=vector_db._collection.count(),  # Get count from Chroma
        )
    except Exception as e:
        print(f"Error building RAG knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build RAG knowledge base: {e}",
        )


@app.post(
    "/quiz/answer", response_model=BatchQuizAnswerItem, status_code=status.HTTP_200_OK
)
async def get_quiz_answer_single(request: BatchQuestionItem):
    """
    Endpoint to get an answer for a single multiple-choice quiz question using RAG.
    """
    if not rag_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Please ensure the backend has started correctly or trigger a build.",
        )

    try:
        predicted_option = answer_quiz_question(request.question, rag_chain)

        cleaned_expected = None
        if isinstance(request.answer, str):
            cleaned_expected = request.answer.strip().upper()

        cleaned_predicted = predicted_option.strip().upper()

        return BatchQuizAnswerItem(
            question=request.question,
            predicted_option=predicted_option,
            expected_answer=cleaned_expected,
            is_correct=cleaned_predicted == cleaned_expected,
        )
    except Exception as e:
        print(f"Error answering single quiz question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get answer for question: {e}",
        )


@app.post(
    "/quiz/batch-answer",
    response_model=BatchQuizAnswerResponse,
    status_code=status.HTTP_200_OK,
)
async def get_quiz_answer_batch(request: BatchQuizRequest):
    """
    Endpoint to get answers for a batch of multiple-choice quiz questions using RAG.
    Can be used for evaluation if 'answer' field is provided in input.
    """
    if not rag_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Please ensure the backend has started correctly or trigger a build.",
        )

    results = []
    correct_predictions = 0
    total_questions = len(request.questions)
    has_expected_answers = False

    for q_item in request.questions:
        try:
            predicted_option = answer_quiz_question(q_item.question, rag_chain)
            is_correct = None

            if q_item.answer is not None:
                has_expected_answers = True
                # Clean and compare: remove whitespace, convert to upper
                cleaned_expected = q_item.answer.strip().upper()
                cleaned_predicted = predicted_option.strip().upper()
                is_correct = cleaned_predicted == cleaned_expected
                if is_correct:
                    correct_predictions += 1

            results.append(
                BatchQuizAnswerItem(
                    question=q_item.question,
                    expected_answer=q_item.answer,
                    predicted_option=predicted_option,
                    is_correct=is_correct,
                )
            )
        except Exception as e:
            print(f"Error processing batch question '{q_item.question[:50]}...': {e}")
            results.append(
                BatchQuizAnswerItem(
                    question=q_item.question,
                    expected_answer=q_item.answer,
                    predicted_option="ERROR",  # Indicate an error occurred for this specific question
                    is_correct=False,  # Mark as incorrect if an error occurred
                )
            )

    accuracy = None
    if has_expected_answers and total_questions > 0:
        accuracy = (correct_predictions / total_questions) * 100.0

    return BatchQuizAnswerResponse(
        total_questions=total_questions,
        correct_predictions=correct_predictions if has_expected_answers else None,
        accuracy=accuracy,
        results=results,
    )


# --- Helper for running locally ---
if __name__ == "__main__":
    import uvicorn

    # This will run the FastAPI app directly.
    # In production, you would use a WSGI server like Gunicorn.
    uvicorn.run(
        "server:app", host="0.0.0.0", port=8000, reload=True
    )  # reload=True for development
