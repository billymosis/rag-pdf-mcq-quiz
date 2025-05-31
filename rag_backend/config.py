# --- Paths ---
PDF_DIRECTORY = "./data/5_estate_planning/Lessons/"
PERSIST_DIRECTORY = "./chroma_db"
QUIZ_QUESTIONS_PATH = "./data/processed/5_estate_planning_questions.json"

# --- Chunking Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RAG Parameters ---
TOP_K_RETRIEVAL = 8  # Number of most relevant documents to retrieve
LLM_MODEL_NAME = "gemini-1.5-flash"
LLM_MODEL_EMBEDDING = "models/text-embedding-004"
# LLM_MODEL_EMBEDDING = "gemini-embedding-exp-03-07"
LLM_TEMPERATURE = 0.0  # Set to 0.0 for deterministic answers
SIMILARITY_THRESHOLD = 0.7
