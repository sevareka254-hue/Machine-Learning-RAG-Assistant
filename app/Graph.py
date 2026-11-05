
from typing import TypedDict, List

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from app.Retrieval import search_documents
from app.generation import create_llm
from langchain_core.prompts import ChatPromptTemplate


# Define the graph state
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


# Node 1: Retrieve documents
def retrieve_node(state: RAGState):

    question = state["question"]

    documents = search_documents(question)

    return {
        "documents": documents
    }


# Node 2: Generate answer
def generate_node(state: RAGState):

    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant that answers questions
        about Machine Learning lectures.

        Answer using ONLY the provided context.

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

    llm = create_llm()

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": response.content
    }


# Build the graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Compile the graph
rag_graph = graph_builder.compile()


def run_rag(question: str):

    result = rag_graph.invoke({
        "question": question
    })

    return {
        "answer": result["answer"],
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in result["documents"]
        ]
    }


if __name__ == "__main__":

    question = "What is Natural Language Processing?"

    result = run_rag(question)

    print("\n===== LANGGRAPH RAG ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")

    for source in result["sources"]:
        print(source)
