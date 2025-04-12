import streamlit as st
from pdf_utils import extract_text_from_pdf, chunk_text_manual, chunk_text_langchain
from embed_utils import load_embedding_model, generate_embeddings, create_faiss_index
from qa_utils import answer_question

st.set_page_config(page_title="Chat With Your PDF", layout="centered")
st.title("📄 Chat With Your PDF")

pdf_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

if pdf_file:
    st.success("✅ PDF uploaded successfully!")

    text = extract_text_from_pdf(pdf_file)
    with st.expander("📜 View Extracted Text"):
        st.text_area("Extracted Content", text, height=300)

    method = st.radio("Choose Chunking Method:", ["LangChain (recommended)", "Manual"])
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    overlap = st.slider("Chunk Overlap", 0, 300, 100)

    chunks = (chunk_text_langchain if method == "LangChain (recommended)" else chunk_text_manual)(
        text, chunk_size, overlap
    )

    st.success(f"✅ {len(chunks)} chunks created.")

    model = load_embedding_model()
    embeddings = generate_embeddings(model, chunks)
    index = create_faiss_index(embeddings)

    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        answer = answer_question(user_question, model, index, chunks)
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
        st.markdown("### 💡 Answer:")
        st.write(answer)

    st.subheader("📜 Chat History")
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['question']}")
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {chat['answer']}")

    st.subheader("📥 Download Chat History")
    if st.session_state.chat_history:
        chat_text = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in st.session_state.chat_history])
        st.download_button("⬇️ Download", chat_text, file_name="chat_history.txt")
