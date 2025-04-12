import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(model, chunks):
    return model.encode(chunks, show_progress_bar=True)

def create_faiss_index(embeddings):
    embedding_array = np.array(embeddings).astype("float32")
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)
    return index
