
from typing import TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.retrieval.hybrid_search import hybrid_search


MAX_RETRIES = 1

llm = OllamaLLM(
    model="gemma3:4b",
    temperature=0,
    num_predict=512,
)


class AgentState(TypedDict):
    original_question: str
    question: str
    documents: list[dict]
    relevant_documents: list[dict]
    retry_count: int
    answer: str


grade_prompt = ChatPromptTemplate.from_template("""
Determine whether this passage contains useful evidence
for answering the user's question.

Question:
{question}

Passage:
{document}

Reply with exactly YES or NO.
Only say YES if the passage contains information that
directly helps answer the question.
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the question as a focused search query for a
Mahabharata passage. Preserve the original meaning.
Include important names and concepts. Return only the query.

Question:
{question}
""")

answer_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using ONLY the evidence below.

Question:
{question}

Evidence:
{context}

Rules:
- Give a clear answer in 2-4 complete sentences.
- Do not invent facts or unsupported interpretations.
- Cite claims using the supplied [Source N] identifiers.
- Use only source identifiers provided in the evidence.
- If the evidence is insufficient, say so.
- Do not copy incomplete sentences from the passages.

IMPORTANT RULES:
- Answer only claims directly supported by the provided passages.
- Do not invent, exaggerate, or distort the meaning of the text.
- Do not turn a passage about battle into a different instruction.
- Cite the source that directly supports each factual claim.
- If the passages do not clearly support an answer, say so.
- Use neutral language and distinguish what the text says from interpretation.
- Identify who is speaking in each passage.
- Never attribute a statement by Bhishma, Vyasa, or another character to Krishna.
- Distinguish direct quotations from your interpretation.
- Use a source only for claims that the passage actually supports.

SOURCE ATTRIBUTION RULES:
- Identify the speaker of each passage before answering.
- Never attribute Bhishma's statements to Krishna.
- Use only sources that directly support each claim.
- Cite the correct source ID for every factual claim.
- Do not combine claims from different passages under one citation.
- If the question asks what Krishna said but the retrieved passages
  contain only Bhishma's statements, explicitly say that distinction.
Answer:
""")


def retrieve_node(state: AgentState) -> dict:
    print("\n[Node] retrieve")
    documents = hybrid_search(state["question"])
    print("Retrieved:", len(documents))
    return {"documents": documents}


def grade_documents_node(state: AgentState) -> dict:
    print("[Node] grade_documents")
    relevant_documents = []

    for document in state["documents"]:
        response = (grade_prompt | llm).invoke({
            "question": state["original_question"],
            "document": document["text"],
        })

        verdict = response.strip().upper()
        metadata = document["metadata"]

        print(
            f"Grade: {verdict[:10]} | "
            f"{metadata.get('book')} — "
            f"{metadata.get('section')}"
        )

        if verdict.startswith("YES"):
            relevant_documents.append(document)

    print("Relevant:", len(relevant_documents))
    return {"relevant_documents": relevant_documents}


def route_after_grading(state: AgentState) -> str:
    if state["relevant_documents"]:
        return "generate"

    if state["retry_count"] < MAX_RETRIES:
        return "retry"

    return "no_evidence"


def retry_node(state: AgentState) -> dict:
    print("[Node] retry — rewriting search query")

    response = (rewrite_prompt | llm).invoke({
        "question": state["original_question"],
    })

    refined_question = response.strip().strip('"')

    if not refined_question:
        refined_question = state["original_question"]

    print("Refined query:", refined_question)

    return {
        "question": refined_question,
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: AgentState) -> dict:
    print("[Node] generate")

    context_parts = []

    for source_id, document in enumerate(
        state["relevant_documents"], start=1
    ):
        metadata = document["metadata"]

        context_parts.append(
            f"[Source {source_id}]\n"
            f"Book: {metadata.get('book')}\n"
            f"Section: {metadata.get('section')}\n"
            f"Pages: {metadata.get('page_start')}-"
            f"{metadata.get('page_end')}\n"
            f"Passage: {document['text']}"
        )

    context = "\n\n".join(context_parts)

    response = (answer_prompt | llm).invoke({
        "question": state["original_question"],
        "context": context,
    })

    return {"answer": response.strip()}


def no_evidence_node(state: AgentState) -> dict:
    print("[Node] no_evidence")

    return {
        "answer": (
            "I couldn't find sufficiently relevant evidence "
            "in the retrieved passages to answer this question reliably."
        )
    }


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("retry", retry_node)
builder.add_node("generate", generate_node)
builder.add_node("no_evidence", no_evidence_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "retry": "retry",
        "no_evidence": "no_evidence",
    },
)

builder.add_edge("retry", "retrieve")
builder.add_edge("generate", END)
builder.add_edge("no_evidence", END)

graph = builder.compile()


if __name__ == "__main__":
    # question = (
    #     "What does Krishna teach Arjuna about "
    #     "work without attachment?"
    # )
    question = "What is the capital of France?"

    result = graph.invoke({
        "original_question": question,
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "answer": "",
    })

    print("\n" + "=" * 60)
    print("QUESTION:", result["original_question"])
    print("RETRIES:", result["retry_count"])
    print("\nANSWER:\n")
    print(result["answer"])

    if result["relevant_documents"]:
        print("\nSOURCES:")

        for source_id, document in enumerate(
            result["relevant_documents"], start=1
        ):
            metadata = document["metadata"]
            print(
                f"[Source {source_id}] "
                f"{metadata.get('book')} | "
                f"{metadata.get('section')} | "
                f"Pages {metadata.get('page_start')}-"
                f"{metadata.get('page_end')}"
            )
