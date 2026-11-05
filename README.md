# Machine Learning Lecture RAG Assistant

A local Retrieval-Augmented Generation (RAG) application that answers
questions about a collection of Machine Learning lecture PDFs. The
project uses **LangChain** for document loading, splitting, embeddings,
retrieval, and prompting; **ChromaDB** for persistent vector storage;
**Ollama** for local embedding and chat models; **LangGraph** to
organize the RAG workflow; and **FastAPI** to expose the system through
an API.

## 1. Project Overview

The application loads lecture PDFs, splits their text into smaller
chunks, converts each chunk into an embedding, and stores the chunks and
metadata in a persistent Chroma vector database. When a user submits a
question, the retriever finds the most relevant chunks. The local
language model generates an answer using the retrieved context and is
instructed to say that it does not know when the context is
insufficient.

### Main technologies

-   Python 3.12
-   LangChain
-   LangGraph
-   ChromaDB (`langchain-chroma`)
-   Ollama
-   `nomic-embed-text` --- embedding model
-   `llama3.2:3b` --- chat model
-   FastAPI and Pydantic
-   PyPDFLoader for PDF ingestion

## 2. Document Collection

The project uses three PDF lecture documents:

-   `ML_lecture1.pdf`
-   `ML2_Lecture2.pdf`
-   `ML2_Lecture3.pdf`

The loader successfully extracted **103 pages** from the three PDFs.

## 3. RAG Workflow

1.  **Ingestion:** `PyPDFLoader` loads PDF pages as LangChain `Document`
    objects.
2.  **Chunking:** `RecursiveCharacterTextSplitter` splits the documents
    using:
    -   Chunk size: `1000` characters
    -   Chunk overlap: `200` characters
    -   Separators: paragraph breaks, line breaks, spaces, and
        individual characters.
3.  **Embeddings:** Ollama's `nomic-embed-text` model converts chunks
    and questions into vectors.
4.  **Vector storage:** Chroma stores the vectors, original text, and
    metadata in the `ml_lectures` collection under `vectorstore/`.
5.  **Retrieval:** Similarity search retrieves the top `3` chunks
    (`TOP_K = 3`).
6.  **Generation:** Ollama's `llama3.2:3b` model generates an answer
    from the retrieved context.
7.  **Graph orchestration:** LangGraph runs the `retrieve` node followed
    by the `generate` node.

## 4. Project Structure

The following is the documented structure of the main project
components. Keep the names aligned with the actual files in your local
project.

``` text
RAG_project/
├── app/
│   ├── ingestion.py       # Load PDF documents
│   ├── chunking.py        # Split documents into chunks
│   ├── embeddings.py      # Create embeddings and Chroma store
│   ├── retrieval.py       # Similarity search
│   ├── generation.py      # Prompt and LLM generation
│   ├── graph.py           # LangGraph workflow
│   └── main.py            # FastAPI application
├── documents/             # Source lecture PDFs
├── vectorstore/           # Persistent Chroma database (generated)
├── .env                   # Local environment settings (not committed)
├── .env.example           # Example environment settings
├── .gitignore
└── requirements.txt
```

## 5. Requirements

-   Windows, macOS, or Linux
-   Python 3.12 recommended
-   Ollama installed and running
-   The following Ollama models:
    -   `nomic-embed-text`
    -   `llama3.2:3b`

Pull the models if they are not already installed:

``` bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## 6. Installation and Setup

Open a terminal in the project root.

### Create and activate a virtual environment (Windows PowerShell)

``` powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install Python dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables

Copy `.env.example` to `.env` and adjust the values if your application
reads configuration from these files. Do not commit `.env` or secret
values. The model names used by the current implementation are
`nomic-embed-text` and `llama3.2:3b`.

## 7. Build the Vector Database

Run the ingestion and chunking modules as needed:

``` bash
python -m app.ingestion
python -m app.chunking
```

Create or rebuild the persistent Chroma vector store:

``` bash
python -m app.embeddings
```

The successful project test stored **104 chunks** in the `ml_lectures`
collection.

> Rebuilding the vector store may add duplicate records if the
> collection is not cleared first. If you need to rebuild, remove or
> reset the existing local `vectorstore/` data before running ingestion
> again.

## 8. Run the Application

Make sure Ollama is running and the required models are available. Start
the FastAPI server from the project root:

``` bash
uvicorn app.main:app --reload
```

Open the interactive Swagger documentation:

-   `http://127.0.0.1:8000/docs`

### Available API endpoints

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  GET                     `/`                     Check that the API is
                                                  running

  GET                     `/health`               Return the health
                                                  status

  POST                    `/ask`                  Submit a question and
                                                  receive an answer with
                                                  source metadata
  -----------------------------------------------------------------------

Example request for `POST /ask`:

``` json
{
  "question": "What is Natural Language Processing?"
}
```

The `/ask` request uses Pydantic validation. The question must contain
between 3 and 1000 characters. A malformed JSON body can return
`422 Unprocessable Entity`.

Example response shape:

``` json
{
  "answer": "Answer generated from the retrieved lecture context.",
  "sources": [
    {
      "source": "documents/ML_lecture1.pdf",
      "page": 8
    }
  ]
}
```

The exact answer and sources depend on the question and retrieval
results.

## 9. Configuration and Troubleshooting

### CUDA out of memory

A CUDA out-of-memory error means the GPU does not have enough available
VRAM for the model operation. On the development machine, running Ollama
with CPU execution succeeded. On Windows, set the user environment
variable:

``` text
CUDA_VISIBLE_DEVICES=-1
```

Then fully quit and restart Ollama and restart the FastAPI server. Use
`ollama ps` to inspect the model processor allocation. CPU execution can
be slower but avoids relying on limited GPU memory.

### Common checks

``` bash
ollama list
ollama ps
```

Ensure the model names in the code match the models installed in Ollama.

## 10. Evaluation Performed

The following checks were performed during development:

  -------------------------------------------------------------------------------------
  Test                                Observed result
  ----------------------------------- -------------------------------------------------
  PDF loading                         103 pages loaded from the three PDFs

  Chunking                            104 chunks created

  Chroma storage                      104 chunks stored successfully

  Retrieval                           Top 3 chunks returned; specific lecture-related
                                      questions produced more relevant results than a
                                      broad question

  In-scope generation                 NLP question produced an answer with lecture
                                      source pages 8, 7, and 11

  Out-of-scope generation             System returned:
                                      `I don't know based on the provided documents.`

  LangGraph                           `retrieve` then `generate` graph executed and
                                      returned an answer and sources

  FastAPI                             `/ask` was tested; a malformed JSON request
                                      returned 422 as expected. CPU execution was used
                                      to work around a CUDA out-of-memory issue.
  -------------------------------------------------------------------------------------

## 11. Limitations

-   Retrieval quality depends on the wording and specificity of the
    question.
-   The current retriever uses similarity search with `k=3`; this value
    may need evaluation and tuning.
-   The language model can only be as reliable as the retrieved context.
    Source metadata is returned to help users verify answers.
-   CPU inference may be slower than GPU inference.
-   This README documents the confirmed routes `/`, `/health`, and
    `/ask`. Add any additional routes only if they exist in the final
    code.

## 12. Security Notes

-   Keep `.env` out of version control.
-   Commit `.env.example` without passwords, tokens, or private keys.
-   Do not upload private lecture material or other restricted documents
    to a public repository unless you have permission.
