# DocuAsk - Stage 2 Interview Task - RAG System

## Introduction

Welcome to Stage 2 of the interview process with DocuAsk!

This repository contains the materials and instructions for a take-home technical assessment. The goal is to evaluate your approach to problem-solving in a generative AI context, 
specifically using **Retrieval Augmented Generation (RAG)**. We're interested in how you think about retrieving information and solving problems in this domain, rather than UI 
development.

This task is based on a real-world requirement from the MFPC (Malaysian Financial Planning Council) involving their educational materials.

## The Task

Your objective is to **build a RAG system that can accurately answer multiple-choice quiz questions** based on the provided financial planning course materials.

The system should:
1.  Consume the content from the provided PDF course materials (details below).
2.  When given a multiple-choice quiz question from the corresponding chapter quiz:
    *   Retrieve relevant information from the PDF content.
    *   Use this retrieved information (and an LLM) to determine the correct multiple-choice answer (e.g., A, B, C, or D).

## Provided Materials

Inside this repository, you will find:

1.  **`data/5_estate_planning/Lessons`**: A folder containing 8 PDF files, representing Chapters 1 through 8 of the "Estate Planning" module. This is the source material your RAG system should 
use.
2.  **`data/processed/5_estate_planning_questions.json`**: A JSON file containing all the multiple-choice questions for the 8 chapters (10 questions per chapter). Each entry includes:
    *   `chapter_key`: Identifier for the chapter (e.g., "chapter1").
    *   `question`: The full text of the multiple-choice question, including the options (A, B, C, D).
    *   `expected_answer`: The correct option (e.g., "A", "B", "C", or "D").

*(Optional: You may want to include the actual quiz PDFs in a `data/5_estate_planning` folder as well for context, even if the primary evaluation uses the JSON).*

## Goals & Success Criteria

*   **Primary Goal:** Achieve the highest possible accuracy in answering the 80 quiz questions using your RAG system.
*   **Minimum Accuracy:** Your system must achieve at least **75%** accuracy across all questions.
*   **Excellent Accuracy:** Your system is *awesome* if it achieves **85%** accuracy across all questions.
*   **(Benchmark):** For reference, an internal implementation achieved 97% accuracy. Aim high!

## Deliverables

Please structure your submission within this repository (or a fork of it). We are looking for two main components:

1.  **RAG API/System:**
    *   The core backend logic for your RAG system.
    *   This could be structured as an API endpoint (e.g., using Flask, FastAPI, Node.js/Express) that accepts a quiz question (from `quiz_data.json`) and returns the predicted 
answer (A, B, C, or D).
    *   This system should implement the logic for processing/chunking the PDFs, embedding, retrieving relevant context, and using an LLM to select the answer based on the context.

2.  **Validator Script:**
    *   A script (e.g., Python, Node.js) that automatically evaluates your RAG system's performance.
    *   This script should:
        *   Read the questions and expected answers from `quiz_data.json`.
        *   Call your RAG API/System for each question.
        *   Compare the predicted answer from your API with the `expected_answer`.
        *   Calculate and print the overall accuracy percentage.
        *   (Highly Recommended): Output details about incorrect answers (e.g., question, expected answer, predicted answer) to help identify patterns and areas for improvement 
during your development.

**Focus:** Please concentrate on the backend RAG pipeline, data processing, retrieval strategy, prompt engineering, and validation. **No UI/frontend development is required.**

## Key Considerations & Hints

*   **PDF Processing:** How will you extract text and potentially structure/chunk the information from the PDFs effectively?
*   **Retrieval Strategy:** How will you ensure the most relevant context is retrieved for each specific quiz question? Consider embedding models and retrieval techniques.
*   **Answer Selection:** LLMs excel at generating text, but multiple-choice requires selecting a specific option. How will you prompt the LLM or structure the process to reliably 
choose A, B, C, or D based *only* on the retrieved context? Be wary of hallucination or the LLM using its general knowledge.
*   **Validation Loop:** Use your validator script iteratively during development to measure improvements and guide your efforts.
*   **Question Nuances:** Pay attention to how the questions are phrased; some might contain subtleties or require careful consideration of the retrieved text.

## Technology Stack

You are free to choose the programming languages, frameworks, and libraries you are most comfortable with (e.g., Python, Langchain, LlamaIndex, Node.js, etc.). You can use any LLM 
accessible via API (like OpenAI's models, Anthropic's Claude, etc.) or run local models if you prefer.

(Feel free to request an API key from gemini if you'd like)

## Next Steps & Communication

1.  Clone or fork this repository.
2.  Review the materials and the task description.
3.  Begin designing and implementing your solution.
4.  **Reach out anytime:** If you have questions or need clarification, please don't hesitate to message Joseph Chin (contact details provided separately, e.g., via email, whatsapp or through the agency). Collaboration and asking questions are encouraged.
5.  **Submission & Review:** Please aim to have your solution ready for discussion by the agreed-upon deadline (typically the assessment should take no more than a few days). Let Joseph know your availability. We will schedule a call to walk through your code, design choices, and results.

---

Good luck! We're looking forward to seeing your approach.
