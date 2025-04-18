import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load embedding model
model = SentenceTransformer("BAAI/bge-m3")

# Load vector store
def load_vector_store(index_path="vector_index.faiss", chunk_path="chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_vector_store()

# Semantic search
def search_similar_chunks(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# Prompt template
def build_prompt(question, context):
    return f"""You are a helpful travel assistant. Use the context below to answer the question. 
If the answer is not in the context, say "I'm not sure based on the available information."

Context:
{context}

Question: {question}
Answer:"""

# Call LLaMA via Ollama
def ask_llama(prompt):
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

# Generate answer using prompt + context
def generate_answer(question, index, chunks):
    relevant_chunks = search_similar_chunks(question, index, chunks)
    context = "\n".join(relevant_chunks)
    prompt = build_prompt(question, context)
    return ask_llama(prompt)

# --------- FastAPI Setup ---------
app = FastAPI()

# Optional: CORS for frontend (like React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = generate_answer(query.question, index, chunks)
    return {"answer": answer}
