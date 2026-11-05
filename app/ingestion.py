
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Get the documents folder
DOCUMENTS_DIR = BASE_DIR / "documents"


def load_documents():
    """Load all PDF files from the documents folder."""

    all_documents = []

    # Find all PDF files
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    # Load each PDF file
    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        all_documents.extend(documents)

        print(f"Loaded {len(documents)} pages")

    print(f"\nTotal pages loaded: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    documents = load_documents()

    print("\nFirst page preview:")
    print(documents[0].page_content[:500])

    print("\nFirst page metadata:")
    print(documents[0].metadata)
