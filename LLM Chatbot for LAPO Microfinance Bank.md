# LLM Chatbot for LAPO Microfinance Bank

This project is an AI-powered chatbot designed to assist users with inquiries related to LAPO Microfinance Bank. The chatbot, named Lala, leverages a sophisticated agent built with LangGraph to provide accurate and context-aware responses. It can access internal documentation, perform real-time web searches, and utilize large language models to answer a wide range of questions about LAPO's products, services, and policies.




## Features

*   **Intelligent Agent (Lala):** Powered by LangGraph, Lala is designed to understand and respond to user queries effectively.
*   **Context-Aware Responses:** The chatbot maintains conversation context to provide relevant and coherent answers.
*   **Internal Document Search (RAG):** Utilizes FAISS for efficient retrieval-augmented generation (RAG) from LAPO Microfinance Bank's internal documents (TXT, PDF, DOCX, HTML, JSON).
*   **Real-time Web Search:** Integrates with EXA AI Search, DuckDuckGo, and Google Programmable Search for up-to-date information, with a fallback mechanism.
*   **General Knowledge (ChatGPT Lookup):** Can answer broader questions related to LAPO Microfinance Bank using a general-purpose LLM.
*   **Scheduled Index Updates:** Automatically refreshes the document index daily to ensure the RAG system has the latest information.
*   **Django Backend:** Provides a robust and scalable web application framework.
*   **RESTful API:** Exposes endpoints for asking questions and summarizing conversation context.




## Installation

To set up and run the LLM Chatbot, follow these steps:

### Prerequisites

*   Python 3.9+
*   Git
*   `pip` (Python package installer)

### Environment Variables

Create a `.env` file in the root directory of the project and add the following environment variables. Obtain the API keys from their respective providers.

```
OPENAI_API_KEY="your_openai_api_key"
GROQ_API_KEY="your_groq_api_key"
EXA_API_KEY="your_exa_api_key"
GOOGLE_API_KEY="your_google_api_key"
GOOGLE_CSE_ID="your_google_custom_search_engine_id"
```

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/LAPO-JuniorDevOps/LLM_Chatbot.git
    cd LLM_Chatbot
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements_utf8.txt
    ```

4.  **Prepare internal documents (optional but recommended for RAG):**

    Place your LAPO Microfinance Bank documents (TXT, PDF, DOCX, HTML, JSON) into the `langgraph_agent/company_docs` directory. The system will automatically build a FAISS index from these documents.

5.  **Run Django migrations:**

    ```bash
    python manage.py migrate
    ```

6.  **Start the Django development server:**

    ```bash
    python manage.py runserver
    ```

    The application will typically be accessible at `http://127.0.0.1:8000/`.

7.  **Access the chatbot:**

    Navigate to `http://127.0.0.1:8000/chat/` in your web browser to interact with the chatbot.




## Usage

Once the server is running, you can interact with the chatbot through the web interface. The chatbot is designed to answer questions about LAPO Microfinance Bank.

### Chatbot Interaction

*   **Ask Questions:** Type your questions into the input field and press Enter or click the send button.
*   **Contextual Conversations:** The chatbot maintains the context of your conversation, allowing for more natural and continuous interactions.
*   **Information Retrieval:** Lala will use its tools to retrieve information from internal documents or the web to answer your queries.

### API Endpoints

The chatbot also exposes API endpoints for programmatic interaction:

*   **`/ask/` (POST):**
    *   **Description:** Sends a query to the chatbot and receives a response.
    *   **Request Body (JSON):**
        ```json
        {
            "query": "Your question here",
            "context": "Previous conversation context (optional)"
        }
        ```
    *   **Response Body (JSON):**
        ```json
        {
            "response": "Chatbot's answer"
        }
        ```

*   **`/summarize/` (POST):**
    *   **Description:** Summarizes a given conversation context.
    *   **Request Body (JSON):**
        ```json
        {
            "previous_summary": "Summary of previous turns (optional)",
            "recent_turns": "Recent conversation turns to summarize"
        }
        ```
    *   **Response Body (JSON):**
        ```json
        {
            "summary": "Summarized conversation"
        }
        ```




## Project Structure

```
LLM_Chatbot/
├── RASA_AI_CHATBOT/             # Contains Rasa-related files (actions, etc.)
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── actions.py           # Custom actions for Rasa
│   │   └── scraper_utils.py     # Utility for web scraping
├── langgraph_agent/             # Core LangGraph agent implementation
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── langgraph_faiss.py       # Main logic for LangGraph agent, FAISS integration, and tool definitions
│   ├── migrations/              # Django database migrations
│   ├── models.py
│   ├── tests.py
│   ├── urls.py                  # URL configurations for the LangGraph agent app
│   └── views.py                 # Django views for API endpoints (ask, summarize)
├── manage.py                    # Django's command-line utility
├── rasabot/                     # Main Django project settings
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings/                # Django settings (base, dev, prod, environment)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dev.py
│   │   ├── environment.py
│   │   └── prod.py
│   ├── urls.py                  # Main URL configurations for the Django project
│   └── wsgi.py
├── rasabotapp/                  # Django app for the chatbot frontend
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── static/                  # Static files (CSS, JS, images) for the frontend
│   │   ├── img/
│   │   └── slides/
│   ├── templates/               # HTML templates for the chatbot interface
│   │   ├── chatbot.html
│   │   ├── chatpage.html
│   │   ├── front.html
│   │   └── index.html
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── staticfiles/                 # Collected static files (Django's `collectstatic`)
├── .env.example                 # Example environment variables file
├── new_README.md                # This documentation file
├── package-lock.json
├── package.json
└── requirements_utf8.txt        # Python dependencies
```




## Technologies Used

*   **Python:** Primary programming language.
*   **Django:** Web framework for the backend.
*   **LangChain:** Framework for developing applications powered by language models.
*   **LangGraph:** Library for building robust and stateful LLM applications.
*   **FAISS:** For efficient similarity search and clustering of dense vectors, used in RAG.
*   **HuggingFace Embeddings:** For generating embeddings for document indexing.
*   **OpenAI API:** For various LLM functionalities (e.g., `gpt-4o-mini`).
*   **Groq API:** For fast inference with language models.
*   **EXA AI Search API:** Primary web search tool for real-time information.
*   **DuckDuckGo Search API:** Fallback web search tool.
*   **Google Programmable Search API:** Final fallback web search tool.
*   **HTML, CSS, JavaScript:** For the frontend web interface.




## Key Contributors/Roles

*   **Supervisor:** Mr. David
*   **Front-end Developers:** Shalom Oluyemi, Wealth Oluyemi
*   **Back-end Developer:** Laura Brusco
*   **Data Scientists/Machine Learning Engineers:** Ogochukwu Nwamata, Victory Osazuwa-Ojo


