import re
from typing import Optional
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

from rag_backend.config import TOP_K_RETRIEVAL


class Search(BaseModel):
    """Search query."""

    query: str = Field(..., description="Broad term search query for this question")
    chapter: str = Field(
        ...,
        description="Which correlated chapter title according to the question based on this [1. Chapter 1  The Concepts and Fundamentals of Estate Planning, 2. Chapter 2  Testacy and Intestacy, 3. Chapter 3  Estate of Muslims, 4. Chapter 4  Trusts, 5. Chapter 5  Powers of Attorney, 6. Chapter 6  Personal Representatives Duties and Powers, 7. Chapter 7  Life Insurance and Estate Planning, 8. Chapter 8  Estate Planning for Business Owners, Estate Planning-13-15] answer based on that array, if you don't know answer N/A",
    )
    objectives: str = Field(
        ...,
        description="Objectives of the question",
    )


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    query: Search
    expected_answer: Optional[str]


def build_rag_chain(vector_db: Chroma, llm: ChatGoogleGenerativeAI):
    """
    Builds and returns a LangChain RetrievalQA chain.
    """
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in estate planning in Malaysia, expert in muslim rules and expert in real estate law including the insurance part, and also at answering multiple-choice questions. Your task is to answer the given multiple-choice question ONLY based on the provided context. If the context does not contain enough information to determine the correct answer, respond with 'N/A'. Your final answer MUST be ONLY the letter (A, B, C, or D) corresponding to the correct option. Do NOT include any additional text, explanations, or punctuation before the letter. You may add a brief explanation *after* the letter if you wish, but the letter must be the first character of your response.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
    )

    # --- Graph Nodes ---

    def analyze_query(state: State):
        structured_llm = llm.with_structured_output(Search)
        if isinstance(state.get("expected_answer"), str):
            pass
        query = structured_llm.invoke(state.get("question"))
        print(query)
        return {"query": query, "expected_answer": state.get("expected_answer")}

    def retrieve(state: State):
        """
        Retrieves documents from the vector store based on the question.
        """
        print("---RETRIEVE NODE---")
        query = state["query"]
        # Use the vector_db directly for similarity search
        retrieved_docs = vector_db.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={
                "k": TOP_K_RETRIEVAL,
                # "score_threshold": 0.8,
                # "filter": {
                #     "$or": [
                #         {"objectives": {"$eq": query.objectives}},
                #         {"chapter": {"$eq": query.chapter}},
                #     ]
                # },
            },
        )
        docs = retrieved_docs.invoke(
            f"{query.query}, possible chapter {query.chapter}, question objectives: {query.objectives}"
        )
        print("chapter: ", query.chapter)
        print(f"retrieved relevant: {len(docs)} docs")
        return {"context": docs}

    def generate(state: State):
        """
        Generates an answer based on the retrieved context and question.
        """
        print("---GENERATE NODE---")
        question = state["question"]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        # Create a chain for generation
        # Pass the formatted prompt and LLM directly
        rag_chain = rag_prompt | llm

        # Invoke the chain with the current state's context and question
        response = rag_chain.invoke({"question": question, "context": docs_content})

        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    return graph


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


def answer_quiz_question(
    question_text: str, rag_chain: RetrievalQA, expected_answer: Optional[str] = None
) -> str:
    """
    Invokes the RAG chain to get an answer for a single quiz question.
    """
    print(f"\nProcessing question: {question_text[:80]}...")

    payload = {"question": question_text}
    if isinstance(expected_answer, str):
        payload["expected_answer"] = expected_answer

    result = rag_chain.invoke(payload)

    predicted_answer_raw = result["answer"].strip()
    predicted_option = extract_predicted_option(predicted_answer_raw)

    print(f"  Predicted raw: {predicted_answer_raw[:50]}...")
    print(f"  Extracted option: {predicted_option}")

    return predicted_option
