
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.chunking import split_documents
from app.ingestion import load_documents


# Embedding model configuration
EMBEDDING_MODEL = "nomic-embed-text"

# Vector database directory
BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def create_embeddings():

    # Initialize the embedding model
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings



def create_vectorstore():

    # Load original documents
    documents = load_documents()

    # Split documents into chunks
    chunks = split_documents(documents)

    # Initialize embedding model
    embeddings = create_embeddings()

    # Delete the old collection before rebuilding
    try:
        old_vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
            collection_name="ml_lectures"
        )

        old_vectorstore.delete_collection()

    except Exception:
        pass

    # Create a new Chroma vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="ml_lectures"
    )

    print("\n===== VECTOR STORE CREATED =====")
    print(f"Total chunks stored: {len(chunks)}")
    print(f"Database location: {VECTORSTORE_DIR}")

    return vectorstore



if __name__ == "__main__":

    vectorstore = create_vectorstore()

    print("\nVector database created successfully!")
