from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import SecretStr

from rag_backend import config
from rag_backend.vector_store_manager import get_gemini_api_key


def get_retriever(vector_db: Chroma) -> ContextualCompressionRetriever:
    """Create enhanced retriever with query expansion and filtering"""
    gemini_api_key = get_gemini_api_key()
    base_retriever = vector_db.as_retriever(
        search_kwargs={"k": config.TOP_K_RETRIEVAL * 2}  # Retrieve more initially
    )

    # Query expansion (generate multiple queries per question)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=SecretStr(gemini_api_key),
        ),
    )

    # Relevance filtering
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.LLM_MODEL_EMBEDDING, google_api_key=SecretStr(gemini_api_key)
    )
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=config.SIMILARITY_THRESHOLD
    )

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=multi_query_retriever
    )
