
import streamlit as st
from src.agent.graph import graph

st.set_page_config(
    page_title="Mahabharata AI",
    page_icon="📜",
    layout="centered",
)

st.title("📜 Mahabharata AI")
st.caption("Ask questions and get answers grounded in Mahabharata passages.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about the Mahabharata...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the text and generating an answer..."):
            try:
                result = graph.invoke({
                    "original_question": question,
                    "question": question,
                    "documents": [],
                    "relevant_documents": [],
                    "retry_count": 0,
                    "answer": "",
                })

                answer = result["answer"]
                st.markdown(answer)

                sources = result.get("relevant_documents", [])
                if sources:
                    with st.expander("View source passages"):
                        for i, source in enumerate(sources, start=1):
                            metadata = source.get("metadata", {})
                            st.markdown(
                                f"**Source {i}** — "
                                f"{metadata.get('book', 'Unknown book')} | "
                                f"{metadata.get('section', 'Unknown section')} | "
                                f"Pages {metadata.get('page_start', '?')}"
                                f"-{metadata.get('page_end', '?')}"
                            )
                            st.write(source.get("text", ""))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
