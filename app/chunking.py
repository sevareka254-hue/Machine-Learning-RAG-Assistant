
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion import load_documents


# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents):

    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    # Load original documents
    documents = load_documents()

    # Split documents
    chunks = split_documents(documents)

    print("\n===== CHUNKING RESULTS =====")

    print(f"Original pages: {len(documents)}")

    print(f"Total chunks: {len(chunks)}")

    print("\n===== FIRST CHUNK =====")

    print(chunks[0].page_content)

    print("\n===== FIRST CHUNK METADATA =====")

    print(chunks[0].metadata)

    print("\n===== CHUNK LENGTH =====")

    print(len(chunks[0].page_content))
