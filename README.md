# 📚 DocuAsk RAG System – MCQ Answering Engine

This project is a backend system that answers multiple-choice questions (MCQs) using a Retrieval-Augmented Generation (RAG) pipeline. It was built as part of a technical assessment for an AI startup and achieved **80%+ accuracy** across a real 80-question dataset.

---

## 🧠 Summary

* **Goal**: Answer financial planning MCQs using only PDF materials (estate planning modules).
* **Method**: Parse → Chunk → Embed → Retrieve → Prompt LLM → Predict answer (A/B/C/D).
* **Result**: 80%+ accuracy on real data from the Malaysian Financial Planning Council (MFPC).
* **API**: Built with **FastAPI**, fully testable via `/quiz/answer` and `/quiz/batch-answer` endpoints.

---

## 🗂️ Project Structure

```
.
├── data/                     # Source PDFs and question sets
│   ├── Lessons/              # Chapter PDFs and parsed .json/.md
│   ├── Quiz/                 # Quiz question PDFs
│   └── processed/            # Final question dataset (.json)
├── rag_backend/              # Core backend: chunking, retrieval, RAG chain
├── chroma_db/                # Persisted vector store
├── validator_script.py       # Evaluates overall accuracy
├── main.py                   # Script-based pipeline interface
├── server.py                 # FastAPI backend server
├── pyproject.toml            # Dependencies
└── README.md
```

---

## 📊 Model Performance Across Versions

| Version | Commit    | Total Questions | Correct | Incorrect | Accuracy   | Model                          | Duration  | Threads |
| ------- | --------- | --------------- | ------- | --------- | ---------- | ------------------------------ | --------- | ------- |
| v0      | `bee88bd` | 80              | 55      | 25        | 68.75%     | gemini-2.5-flash-preview-05-20 | –         | 1       |
| v1      | `7514868` | 80              | 49      | 31        | 61.25%     | gemini-2.5-pro-preview-05-06   | 4330.68 s | 1       |
| v2      | `1991ae8` | 80              | 52      | 28        | 65.00%     | gemini-1.5-flash               | 14.96 s   | 8       |
| v3      | `03b2a3e` | 80              | 50      | 30        | 62.50%     | gemini-1.5-flash               | 59.12 s   | 8       |
| v4      | `1991ae8` | 80              | 25      | 55        | 31.25%     | gemini-2.5-flash-preview-05-20 | 201.17 s  | 8       |
| v5      | `89e0b5b` | 80              | 59      | 21        | 73.75%     | gemini-2.0-flash               | 29.23 s   | 8       |
| v6      | `2593deb` | 80              | 65      | 15        | **81.25%** | gemini-2.0-flash               | 18.41 s   | 8       |
| v7      | `78799a9` | 80              | 64      | 16        | 80.00%     | gemini-2.0-flash               | 20.13 s   | 8       |
| v8      | `8f59e61` | 80              | 58      | 22        | 72.50%     | gemini-2.0-flash               | 25.88 s   | 8       |

📄 **Full log** with all question-level details:
[Google Sheet – Evaluation Results](https://docs.google.com/spreadsheets/d/1WWdc07vQhKObIAzf8ZKCP0wK5rC0b-QBIDYvVjV8FzE/edit?usp=sharing)

---

## 🧪 Features

* 🔍 **Context Retrieval**: Embeds and retrieves PDF content using `ChromaDB`.
* 🧠 **LLM Reasoning**: Uses **Google Gemini** (via `langchain-google-genai`) to answer quiz questions.
* 📄 **High-Fidelity PDF Parsing**: Uses Meta’s `llama-parser` for reliable content structure.
* ⚙️ **FastAPI Server**: Easily testable endpoints for real-time prediction and batch evaluation.
* 📊 **Auto Evaluation**: Script to validate predictions against expected answers.

---

## 🔧 Tech Stack

| Area           | Tools / Libraries                |
| -------------- | -------------------------------- |
| Language       | Python 3.12                      |
| Backend        | FastAPI                          |
| Retrieval      | LangChain + ChromaDB             |
| LLM            | Google Gemini (via API)          |
| PDF Parsing    | llama-parser, pdftext, pypdfium2 |
| OCR (fallback) | rapidocr-onnxruntime             |
| Evaluation     | scikit-learn                     |
| Environment    | dotenv, Pydantic                 |

---

## 🔌 FastAPI Endpoints

### ✅ Root

```http
GET /
```

Returns welcome message.

---

### 🔄 Build or Rebuild Vector DB

```http
POST /rag/build
```

```json
{
  "pdf_directory": "data/5_estate_planning/Lessons",
  "rebuild_db": true
}
```

---

### ❓ Answer a Single Quiz Question

```http
POST /quiz/answer
```

```json
{
  "question": "Which of the following is not true about estate planning? A. ..., B. ..., C. ..., D. ..."
}
```

Returns predicted option (`A`, `B`, `C`, or `D`).

---

### 📦 Batch Answer with Evaluation

```bash
curl -X POST "http://localhost:8000/quiz/batch-answer" \
-H "Content-Type: application/json" \
-d '{
  "questions": [
    {
      "question": "...",
      "answer": "B"
    },
    {
      "question": "...",
      "answer": "D"
    }
  ]
}'
```

Returns accuracy, prediction, and correctness for each question.

---

## 🧠 Design Notes

* ✅ **RAG Pipeline**: Combines LangChain retriever with custom prompt for Gemini.
* ✅ **Chunking Strategy**: Optimized with `llama-parser` to preserve layout structure.
* ✅ **Validation**: Tracks accuracy and highlights failed questions for debugging.
* ✅ **Stateless Server**: API initializes vector DB and LLM on startup for quick querying.

---

## 📎 Context

This system was built for a **real-world technical assessment** by [DocuAsk.ai](https://github.com/chin-jlyc/rag-mcq-tech-test), based on content from the **Malaysian Financial Planning Council (MFPC)**.

---

💬 Contact & Credits

Original task and assessment design by Joseph Chin – DocuAsk.

Implementation, accuracy testing, and improvements by me.

---

## 🙋‍♂️ Author

**Billy Mosis Priambodo**
🔗 [billymosis.com](https://billymosis.com) | [GitHub](https://github.com/billymosis) | [LinkedIn](https://linkedin.com/in/billymosis)
