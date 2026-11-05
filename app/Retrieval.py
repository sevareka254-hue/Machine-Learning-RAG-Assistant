
from langchain_chroma import Chroma

from app.embeddings import (
    create_embeddings,
    VECTORSTORE_DIR
)


# Number of relevant chunks to retrieve
TOP_K = 3


def load_vectorstore():

    # Initialize the embedding model
    embeddings = create_embeddings()

    # Load the existing Chroma database
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name="ml_lectures"
    )

    return vectorstore


def search_documents(question, k=TOP_K):

    # Load the vector database
    vectorstore = load_vectorstore()

    # Search for the most relevant chunks
    results = vectorstore.similarity_search(
        query=question,
        k=k
    )

    return results


if __name__ == "__main__":

    question = "What are Word Embeddings?"
    results = search_documents(question)

    print("\n===== RETRIEVAL RESULTS =====")

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for i, doc in enumerate(results, start=1):

        print(f"\n===== RESULT {i} =====")

        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))

        print("\nContent:")
        print(doc.page_content)
