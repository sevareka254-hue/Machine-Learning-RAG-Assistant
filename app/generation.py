
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.Retrieval import search_documents


# LLM configuration
LLM_MODEL = "llama3.2:1b"


def create_llm():

    # Initialize the local language model
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    return llm


def generate_answer(question):

    # Retrieve relevant documents
    retrieved_docs = search_documents(question)

    # Combine retrieved content into one context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant that answers questions
        about Machine Learning lectures.

        Answer the question using ONLY the context below.

        If the context does not contain enough information,
        say: I don't know based on the provided documents.

        Do not use outside knowledge.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Initialize the LLM
    llm = create_llm()

    # Create the chain
    chain = prompt | llm

    # Generate the answer
    response = chain.invoke({
        "context": context,
        "question": question
    })

    # Return answer and source documents
    return {
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in retrieved_docs
        ]
    }


if __name__ == "__main__":

    question ="What is the capital of Japan?"

    result = generate_answer(question)

    print("\n===== RAG ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")

    for source in result["sources"]:
        print(source)
