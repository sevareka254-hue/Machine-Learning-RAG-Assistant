
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

from pydantic import BaseModel, Field

from app.Graph import run_rag
from app.embeddings import create_vectorstore
from app.ingestion import DOCUMENTS_DIR


# Create FastAPI application
app = FastAPI(
    title="Machine Learning RAG Assistant",
    description="Ask questions about Machine Learning lectures.",
    version="1.0.0"
)


# Request model
class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000
    )


# Response models
class Source(BaseModel):
    source: str
    page: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    message: str
    filename: str


class EmbeddingResponse(BaseModel):
    message: str
    chunks_stored: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Machine Learning RAG API is running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Ask endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    try:
        result = run_rag(request.question)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Upload PDF endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):

    try:

        # Check the file name
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File name is missing"
            )

        # Extract the safe file name
        original_name = Path(file.filename).name

        # Allow PDF files only
        if Path(original_name).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # Read uploaded file
        content = await file.read()

        # Check empty file
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Limit file size to 20 MB
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 20 MB"
            )

        # Basic PDF signature check
        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )

        # Create documents folder if needed
        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Generate a unique file name
        filename = f"{uuid4().hex[:8]}_{original_name}"

        # Save the PDF
        file_path = DOCUMENTS_DIR / filename

        file_path.write_bytes(content)

        return {
            "message": "PDF uploaded successfully",
            "filename": filename
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await file.close()


# Embeddings endpoint
@app.post("/embed", response_model=EmbeddingResponse)
def create_embeddings_endpoint():

    try:

        # Rebuild the vector database
        vectorstore = create_vectorstore()

        # Count stored chunks
        chunks_count = len(
            vectorstore.get()["ids"]
        )

        return {
            "message": "Embeddings created successfully",
            "chunks_stored": chunks_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
