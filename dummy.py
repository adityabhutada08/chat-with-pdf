# import streamlit as st
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline

# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")


# # -------------------------------------
# # Function to extract text from PDF
# # -------------------------------------
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
#             for page_num, page in enumerate(doc, start=1):
#                 page_text = page.get_text()
#                 text += f"\n\n--- Page {page_num} ---\n{page_text}"
#         return text
#     except Exception as e:
#         return f"Error reading PDF: {e}"

# # -------------------------------------
# # Method 1: Manual text chunking
# # -------------------------------------
# def chunk_text_manual(text, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk.strip())
#         start += chunk_size - overlap
#     return chunks

# # -------------------------------------
# # Method 2: LangChain Recursive Chunking
# # -------------------------------------
# def chunk_text_langchain(text, chunk_size=500, chunk_overlap=100):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", "!", "?", " "]
#     )
#     return splitter.split_text(text)

# # -------------------------------------
# # Streamlit UI
# # -------------------------------------
# st.set_page_config(page_title="Chat With Your PDF", layout="centered")
# st.title("📄 Chat With Your PDF")
# st.subheader("Step 1: Upload PDF and Extract Text")
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# # Load sentence transformer model (only once)
# @st.cache_resource
# def load_embedding_model():
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     return model

# if pdf_file is not None:
#     st.success("✅ PDF uploaded successfully!")


#     # Step 1: Extract text from PDF
#     extracted_text = extract_text_from_pdf(pdf_file)

#     # Optional: Show extracted text
#     with st.expander("📜 View Extracted Text"):
#         st.text_area("Extracted Content", extracted_text, height=300)


#     # Step 2: Chunking
#     st.subheader("Step 2: Chunk the Extracted Text")

#     # Choose method for chunking
#     method = st.radio("Choose Chunking Method:", ["LangChain (recommended)", "Manual"])

#     # Set chunk size and overlap
#     chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500)
#     overlap = st.slider("Chunk Overlap (characters)", 0, 300, 100)

#     if method == "LangChain (recommended)":
#         chunks = chunk_text_langchain(extracted_text, chunk_size, overlap)
#     else:
#         chunks = chunk_text_manual(extracted_text, chunk_size, overlap)

#     st.success(f"✅ Chunking complete! {len(chunks)} chunks created.")

#     # Show first 5 chunks
#     st.subheader("🔍 Preview of Chunks:")
#     for i, chunk in enumerate(chunks[:5]):
#         st.markdown(f"**Chunk {i+1}:**")
#         st.write(chunk)


#      # Step 3: Create Embeddings for the Chunks
#     st.subheader("Step 3: Generate Embeddings")
#     model = load_embedding_model()
    
#     with st.spinner("🔄 Generating embeddings..."):
#         embeddings = model.encode(chunks, show_progress_bar=True)
    
#     st.success("✅ Embeddings generated successfully!")
#     st.write(f"Each chunk is now represented as a {len(embeddings[0])}-dimensional vector.")


    
#     # Step 4: Store Embeddings in FAISS
#     st.subheader("Step 4: Store Embeddings and Prepare for Search")

#     # Convert list of vectors to a numpy array of type float32
#     embedding_array = np.array(embeddings).astype("float32")

#     # Create FAISS index
#     dimension = embedding_array.shape[1]  # typically 384 for all-MiniLM-L6-v2
#     index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
#     index.add(embedding_array)  # Add all embeddings to the index

#     st.success(f"✅ FAISS index created with {index.ntotal} vectors!")


#     # Step 5: Ask Questions and Get Answers from LLM
#     st.subheader("Step 5: Ask Questions to Your PDF 📚🤖")

#     user_question = st.text_input("Ask a question based on the PDF:")

#     if user_question:
#         # 1. Embed the user question
#         question_embedding = model.encode([user_question])

#         # 2. Retrieve top-k similar chunks from FAISS
#         k = 3
#         D, I = index.search(np.array(question_embedding).astype("float32"), k)
#         relevant_chunks = [chunks[i] for i in I[0]]

#         # 3. Build the prompt
#         context = "\n\n".join(relevant_chunks)
#         prompt = f"""You are an intelligent assistant. Use the below context to answer the user's question.
        
#         Context:
#         {context}
        
#         Question: {user_question}
#         Answer:"""

#         # 4. Generate answer using LLM
#         st.spinner("💬 Generating answer...")
#         qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
#         result = qa_pipeline(prompt, max_new_tokens=50)[0]["generated_text"]
        
#         # 5. Display result
#         st.markdown("### 🧠 Answer:")
#         st.write(result.split("Answer:")[-1].strip())




















# import streamlit as st
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline

# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# # ------------------------
# # Function: PDF Text Extractor
# # ------------------------
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
#             for page_num, page in enumerate(doc, start=1):
#                 page_text = page.get_text()
#                 text += f"\n\n--- Page {page_num} ---\n{page_text}"
#         return text
#     except Exception as e:
#         return f"Error reading PDF: {e}"

# # ------------------------
# # Function: Manual Chunking
# # ------------------------
# def chunk_text_manual(text, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk.strip())
#         start += chunk_size - overlap
#     return chunks

# # ------------------------
# # Function: LangChain Chunking
# # ------------------------
# def chunk_text_langchain(text, chunk_size=500, chunk_overlap=100):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", "!", "?", " "]
#     )
#     return splitter.split_text(text)

# # ------------------------
# # Load Sentence Transformer
# # ------------------------
# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # ------------------------
# # Streamlit App Starts
# # ------------------------
# st.set_page_config(page_title="Chat With Your PDF", layout="centered")
# st.title("📄 Chat With Your PDF")

# st.markdown("Upload your PDF, ask questions, and get answers using a smart AI assistant!")

# pdf_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

# # Initialize Chat History
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Add Clear Chat Button
# if st.button("🧹 Clear Chat"):
#     st.session_state.chat_history = []

# if pdf_file is not None:
#     st.success("✅ PDF uploaded successfully!")

#     # Step 1: Extract text
#     extracted_text = extract_text_from_pdf(pdf_file)
#     with st.expander("📜 View Extracted Text"):
#         st.text_area("Extracted Content", extracted_text, height=300)

#     # Step 2: Chunk
#     st.subheader("✂️ Step 2: Text Chunking")
#     method = st.radio("Choose Chunking Method:", ["LangChain (recommended)", "Manual"])
#     chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500)
#     overlap = st.slider("Chunk Overlap", 0, 300, 100)

#     chunks = (chunk_text_langchain if method == "LangChain (recommended)" else chunk_text_manual)(
#         extracted_text, chunk_size, overlap
#     )

#     st.success(f"✅ {len(chunks)} chunks created.")
#     for i, chunk in enumerate(chunks[:5]):
#         st.markdown(f"**Chunk {i+1}:**")
#         st.write(chunk)

#     # Step 3: Embedding
#     st.subheader("🔗 Step 3: Create Embeddings")
#     model = load_embedding_model()

#     with st.spinner("🔄 Generating embeddings..."):
#         embeddings = model.encode(chunks, show_progress_bar=True)

#     st.success("✅ Embeddings generated!")

#     # Step 4: FAISS Indexing
#     st.subheader("📦 Step 4: Store Embeddings")
#     embedding_array = np.array(embeddings).astype("float32")
#     dimension = embedding_array.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embedding_array)
#     st.success(f"✅ FAISS index created with {index.ntotal} vectors.")

#     # Step 5–6: Ask Questions
#     st.subheader("🧠 Step 5: Ask Your Question")
#     user_question = st.text_input("Type your question below:")

#     if user_question:
#         with st.spinner("💬 Thinking..."):
#             question_embedding = model.encode([user_question])
#             D, I = index.search(np.array(question_embedding).astype("float32"), k=3)
#             relevant_chunks = [chunks[i] for i in I[0]]
#             context = "\n\n".join(relevant_chunks)

#             prompt = f"""You are an intelligent assistant. Use the below context to answer the user's question.

# Context:
# {context}

# Question: {user_question}
# Answer:"""

#             try:
#                 result = qa_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
#                 final_answer = result.split("Answer:")[-1].strip() or "❗ The model did not return a clear answer."
#             except Exception as e:
#                 final_answer = f"⚠️ Error generating answer: {e}"

#         # Show answer
#         st.markdown("### 💡 Answer:")
#         st.write(final_answer)

#         # Step 7: Maintain Chat History
#         st.session_state.chat_history.append({
#             "question": user_question,
#             "answer": final_answer
#         })

#     # Display Chat History
#     st.subheader("📜 Chat History")
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.markdown(f"**You:** {chat['question']}")
#         with st.chat_message("assistant"):
#             st.markdown(f"**Bot:** {chat['answer']}")

#     # Step 8: Export chat history as .txt
#     st.subheader("📥 Download Chat History")
#     if st.session_state.chat_history:
#         chat_text = "\n\n".join([f"Q: {chat['question']}\nA: {chat['answer']}" for chat in st.session_state.chat_history])
#         st.download_button("⬇️ Download Chat History", chat_text, file_name="chat_history.txt")
