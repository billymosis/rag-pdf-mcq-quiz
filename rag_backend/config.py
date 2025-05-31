# --- Paths ---
PDF_DIRECTORY = "./data/5_estate_planning/Lessons/"
PERSIST_DIRECTORY = "./chroma_db"
QUIZ_QUESTIONS_PATH = "./data/processed/5_estate_planning_questions.json"

# --- Chunking Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RAG Parameters ---
TOP_K_RETRIEVAL = 5  # Number of most relevant documents to retrieve
LLM_MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"  # Or "models/gemini-pro"
LLM_TEMPERATURE = 0.0  # Set to 0.0 for deterministic answers
