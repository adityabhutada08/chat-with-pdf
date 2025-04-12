from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def answer_question(user_question, model, index, chunks, k=3):
    question_embedding = model.encode([user_question])
    D, I = index.search(question_embedding.astype("float32"), k)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are an intelligent assistant. Use the below context to answer the user's question.

Context:
{context}

Question: {user_question}
Answer:"""

    try:
        result = qa_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
        return result.split("Answer:")[-1].strip() or "⚠️ No clear answer."
    except Exception as e:
        return f"⚠️ Error: {e}"
