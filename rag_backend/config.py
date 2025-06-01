# --- Paths ---
PDF_DIRECTORY = "./data/5_estate_planning/Lessons/"
PERSIST_DIRECTORY = "./chroma_db"
QUIZ_QUESTIONS_PATH = "./data/processed/5_estate_planning_questions.json"

# --- Chunking Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RAG Parameters ---
TOP_K_RETRIEVAL = 5  # Number of most relevant documents to retrieve
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_MODEL_EMBEDDING = "models/embedding-001"
LLM_TEMPERATURE = 0.0
