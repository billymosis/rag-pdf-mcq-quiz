import re
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_backend import config


def build_rag_chain(vector_db: Chroma, llm: ChatGoogleGenerativeAI):
    """
    Builds and returns a LangChain RetrievalQA chain.
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVAL})

    # PROMPT ENGINEERING: Refined prompt for better answer extraction and guidance
    template = """You are an expert in estate planning in Malaysia, expert in muslim rules and expert in real estate law including the insurance part, and also at answering multiple-choice questions.
    Your task is to answer the given multiple-choice question ONLY based on the provided context.
    If the context does not contain enough information to determine the correct answer, respond with "N/A".
    Your final answer MUST be ONLY the letter (A, B, C, or D) corresponding to the correct option.
    Do NOT include any additional text, explanations, or punctuation before the letter.
    You may add a brief explanation *after* the letter if you wish, but the letter must be the first character of your response.

    Context:
    {context}

    Question: {question}

    Answer:"""

    RAG_PROMPT_TEMPLATE = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT_TEMPLATE},
    )
    return qa_chain


def extract_predicted_option(llm_response_raw: str) -> str:
    """
    Extracts the predicted option (A, B, C, or D) from the LLM's raw response.
    This function is robust to common variations in LLM output.
    """
    if not llm_response_raw:
        return "N/A"

    # Remove any leading/trailing whitespace
    response_stripped = llm_response_raw.strip()

    # Case 1: Exactly one of A, B, C, D
    if response_stripped.upper() in ["A", "B", "C", "D"]:
        return response_stripped.upper()

    # Case 2: Starts with A, B, C, or D followed by punctuation or space (e.g., "A:", "B.", "C Explanation")
    match = re.match(r"^[ABCDabcd][.:\s]*", response_stripped)
    if match:
        extracted_char = match.group(0)[
            0
        ].upper()  # Get the first character and uppercase it
        if extracted_char in ["A", "B", "C", "D"]:
            return extracted_char

    # Case 3: "I don't know" or similar explicit statements
    if "I don't know" in response_stripped.lower():
        return "N/A"  # Or specific "I_DONT_KNOW" if you want to distinguish

    # Fallback: If nothing specific is found, return N/A
    return "N/A"


def answer_quiz_question(question_text: str, rag_chain: RetrievalQA) -> str:
    """
    Invokes the RAG chain to get an answer for a single quiz question.
    """
    print(f"\nProcessing question: {question_text[:80]}...")

    result = rag_chain.invoke({"query": question_text})

    predicted_answer_raw = result["result"].strip()
    predicted_option = extract_predicted_option(predicted_answer_raw)

    print(f"  Predicted raw: {predicted_answer_raw[:50]}...")
    print(f"  Extracted option: {predicted_option}")

    return predicted_option
