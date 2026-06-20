
import streamlit as st
import PyPDF2
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY_HERE")

model_gemini = genai.GenerativeModel("gemini-2.5-flash")

st.title("📄 Document Question Answering System")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF Uploaded Successfully!")

    # Chunking
    chunks = []

    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    st.write("Number of Chunks:", len(chunks))

    # Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    st.write("Embedding Shape:", embeddings.shape)

    # FAISS Index
    embedding_array = np.array(embeddings).astype("float32")

    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_array)

    st.success("FAISS Index Created Successfully!")

    question = st.text_input("Ask a Question")

    if question:

        query_embedding = model.encode([question]).astype("float32")

        distances, indices = index.search(query_embedding, k=1)

        context = chunks[indices[0][0]]
        prompt = f"""
        
        Answer the question using only the context below.

        Context:
        {context}
        Question:
        {question}
        """

        response = model_gemini.generate_content(prompt)

        st.subheader("AI Answer")

        st.write(response.text)

       
  
