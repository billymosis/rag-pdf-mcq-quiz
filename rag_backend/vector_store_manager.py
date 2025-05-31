import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import SecretStr
from rag_backend import config

# Load environment variables if this script is run standalone
load_dotenv()


def get_gemini_api_key():
    """Retrieves GEMINI_API_KEY from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. Please set it in your .env file."
        )
    return api_key


def create_or_load_vector_store(
    chunks: list = [], persist_directory: str = config.PERSIST_DIRECTORY
):
    """
    Creates embeddings for the given text chunks and stores them in a ChromaDB,
    or loads an existing ChromaDB.
    """
    gemini_api_key = get_gemini_api_key()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.LLM_MODEL_EMBEDDING,
        google_api_key=SecretStr(gemini_api_key),
        task_type="retrieval_document",
    )

    # Check if the database already exists and has documents
    if (
        os.path.exists(persist_directory)
        and os.path.isdir(persist_directory)
        and len(os.listdir(persist_directory)) > 0
    ):
        print(f"Loading existing ChromaDB from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
        print(f"Loaded database with {vectorstore._collection.count()} entries.")
    else:
        if not chunks:
            raise ValueError("Chunks must be provided to create a new vector store.")
        print(f"Creating new ChromaDB at {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB created with {vectorstore._collection.count()} entries.")

    return vectorstore


def initialize_gemini_llm(
    model_name: str = config.LLM_MODEL_NAME, temperature: float = config.LLM_TEMPERATURE
):
    """Initializes and returns a ChatGoogleGenerativeAI LLM instance."""
    gemini_api_key = get_gemini_api_key()
    print(f"Initializing Gemini LLM ({model_name})...")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=gemini_api_key,
    )
    print("Gemini LLM initialized.")
    return llm
