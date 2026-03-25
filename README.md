# Enterprise Loan & Credit Assistant

This repository contains the source code for an autonomous AI-powered assistant designed for enterprise loan and credit risk management. The system intelligently handles user conversations, from initial policy questions to final loan application decisions, by orchestrating a series of specialized agents.

It features a robust backend built with FastAPI, an interactive frontend with Streamlit, and a sophisticated agentic workflow powered by LangChain and Ollama.

## Architecture

The system is built on a modular, agent-based architecture where each component has a specific responsibility. An orchestrator directs the flow of information between these agents, ensuring a logical and efficient process.

  *This is a conceptual representation; the image link is a placeholder to illustrate where a diagram would go.*

1.  **Frontend (Streamlit):** A web-based user interface for interacting with the assistant. It includes features for real-time chat, document uploads, session management, and a "Glass Brain" view to trace the agent's decision-making process.

2.  **Backend (FastAPI):** A high-performance API server that exposes endpoints for chat, RAG queries, and document uploads. It includes rate limiting for stability.

3.  **Orchestrator:** The central component that manages the entire conversation flow. It receives user messages, routes them through the appropriate agents based on intent, and synthesizes the final response.

4.  **Autonomous Agents:**
    *   **Guardrails Agent:** The first and last line of defense. It scans all incoming and outgoing text for security threats (prompt injection, PII), harmful content, and off-topic queries using both LLM-based analysis and regex. It redacts sensitive information and can block inappropriate messages.
    *   **Intake Agent:** Uses an LLM to perform intelligent entity extraction (e.g., "5-year period" -> `tenure_months: 60`) and classifies the user's intent (e.g., `policy_question`, `loan_application`, `calculation`).
    *   **Retrieval Agent (RAG):** Answers policy-related questions by searching a vector database (ChromaDB) containing the bank's master policy documents.
    *   **Tool Agent:** Executes a suite of financial tools to perform calculations like EMI, eligibility checks, and risk scoring.
    *   **Decision Agent:** The final decision-maker. It synthesizes inputs from the Tool Agent, RAG Agent, and conversation history to generate a context-aware, reasoned response and determine the loan application's status (e.g., `APPROVE`, `REJECT`, `NEED_MORE_INFO`).

5.  **Memory Store:** A persistent session storage system using SQLite to maintain conversation state (collected entities, chat summary) across multiple interactions, preventing data loss on server restarts.

6.  **Knowledge Base (RAG):**
    *   **Ingestion:** A script parses the `master_policy_doc.txt` into chunks.
    *   **Storage:** The chunks are converted into vector embeddings using `all-MiniLM-L6-v2` and stored in a ChromaDB database.
    *   **Retrieval:** The `Retrieval Agent` performs similarity searches on this database to find relevant policy excerpts.

7.  **Financial Tools:** A set of deterministic Python modules for core business logic:
    *   `emi_calculator.py`: Calculates Equated Monthly Installment.
    *   `eligibility.py`: Checks basic age and income criteria.
    *   `risk_scoring.py`: Determines the applicant's risk band based on DTI and CIBIL score.

## Key Features

*   **Autonomous Agent Workflow:** A multi-agent system that collaborates to understand, process, and respond to user queries.
*   **Intent-Based Routing:** Intelligently directs user queries to the correct workflow (RAG, Tools, or Data Collection) based on their intent.
*   **Retrieval-Augmented Generation (RAG):** Answers questions based on a private knowledge base, ensuring factual and policy-aligned responses.
*   **Financial Tool Integration:** Executes real-world financial calculations for EMI, eligibility, and risk assessment.
*   **LLM-Powered Security Guardrails:** Provides robust protection against prompt injection, PII leaks, and harmful content.
*   **Persistent Session Memory:** Maintains conversation context across multiple requests using a SQLite database.
*   **"Glass Brain" Traceability:** The UI provides a step-by-step trace of the agents' actions and data, offering full transparency into the decision-making process.
*   **Dynamic Knowledge Updates:** Allows users to upload new documents, which are automatically ingested into the RAG knowledge base.

## Tech Stack

*   **Backend:** FastAPI, Uvicorn
*   **Frontend:** Streamlit
*   **LLM & Orchestration:** LangChain, Ollama (with `mistral` model)
*   **Vector Database:** ChromaDB
*   **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
*   **Core Libraries:** Pydantic, Requests, python-dotenv

## Setup and Installation

### Prerequisites

*   Python 3.9+
*   [Ollama](https://ollama.com/) installed and running.

### How to Run

1.  **Set up Ollama Model:**
    Pull the required model for the agents to use.
    ```sh
    ollama pull mistral
    ```

2.  **Clone the Repository:**
    ```sh
    git clone https://github.com/rakshitmishra89/loan-assistant.git
    cd loan-assistant
    ```

3.  **Install Dependencies:**
    Create a virtual environment and install the required packages.
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4.  **Ingest Knowledge Base:**
    Run the ingestion script to populate the ChromaDB vector store with the policy document.
    ```sh
    python rag/ingest.py
    ```
    This will create the `rag/chroma_db` directory.

5.  **Start the Backend API:**
    Run the FastAPI server using Uvicorn.
    ```sh
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`.

6.  **Start the Frontend UI:**
    In a new terminal, run the Streamlit application.
    ```sh
    streamlit run frontend/app.py
    ```
    The UI will be accessible at `http://localhost:8501`.

## Project Structure

```
.
├── backend/          # FastAPI server, agents, orchestrator, and API logic
│   ├── adapters/     # Bridges between the orchestrator and external modules
│   ├── agents/       # The core autonomous agents (Decision, Intake, etc.)
│   ├── main.py       # FastAPI application endpoints
│   ├── orchestrator.py # Manages the agentic workflow
│   └── memory_store.py # SQLite-based session management
├── frontend/         # Streamlit UI application and components
│   ├── app.py        # Main Streamlit application
│   └── ui_components.py # Reusable UI cards and panels
├── guardrails/       # Security agent for content moderation and PII redaction
│   ├── guardrails.py # Core moderation and intent analysis logic
│   └── tests.json    # Test cases for the guardrails
├── perf/             # Scripts for performance and stress testing
├── rag/              # RAG pipeline, document ingestion, and vector store
│   ├── ingest.py     # Script to populate the vector database
│   ├── retriever.py  # Functionality to query ChromaDB
│   └── chroma_db/    # Persistent ChromaDB vector store
├── tools/            # Standalone financial calculation modules
└── requirements.txt  # Python dependencies
