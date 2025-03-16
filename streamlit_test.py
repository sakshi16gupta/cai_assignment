import yfinance as yf
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re
from transformers import AutoTokenizer

def download_financials(ticker):
    """Download financial statements for a given company ticker."""
    stock = yf.Ticker(ticker)

    # Get financial statements
    income_stmt = stock.financials.T  # Transpose for easier handling
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T

    return income_stmt, balance_sheet, cash_flow

def clean_data(df):
    """Clean and structure the financial data."""
    df = df.fillna(0)  # Fill missing values with 0
    df = df.astype(float)  # Ensure numeric values
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    return df

def validate_query(query):
    """Validate and filter user queries to prevent irrelevant/harmful inputs."""
    forbidden_patterns = [r'\b(hack|attack|fraud|scam)\b', r'[^a-zA-Z0-9 ?!]']
    for pattern in forbidden_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    return True

def convert_to_chunks(text, chunk_size=128):
    """Split text into manageable chunks using a pre-trained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # BERT tokenizer
    tokens = tokenizer.tokenize(text)
    
    # Create chunks using step iteration
    chunks = [tokenizer.convert_tokens_to_string(tokens[i : i + chunk_size]) 
              for i in range(0, len(tokens), chunk_size)]
    
    return chunks

def embed_text(chunks, model):
    """Generate embeddings for text chunks."""
    return model.encode(chunks, convert_to_numpy=True)

def store_embeddings(embeddings):
    """Store embeddings in a FAISS vector database."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_similar(query, model, index, bm25, chunks, top_k=5):
    """Retrieve relevant text chunks using FAISS and BM25."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    faiss_results = [chunks[i] for i in indices[0]]

    # BM25 retrieval
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_results = [chunks[i] for i in bm25_top_indices]

    return list(set(faiss_results + bm25_results))

def rerank_results(query, results, cross_encoder):
    """Re-rank results using a cross-encoder."""
    pairs = [[query, result] for result in results]
    scores = cross_encoder.predict(pairs)
    sorted_results = [x for _, x in sorted(zip(scores, results), reverse=True)]
    return sorted_results, scores

def filter_misleading_responses(results, threshold=0.2):
    """Filter responses with low confidence scores to remove hallucinated/misleading answers."""
    return [res for res in results if res[1] >= threshold]

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Streamlit UI
st.title("Financial Data Q&A")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if ticker:
    income_stmt, balance_sheet, cash_flow = download_financials(ticker)
    
    # Clean data
    income_stmt = clean_data(income_stmt)
    balance_sheet = clean_data(balance_sheet)
    cash_flow = clean_data(cash_flow)
    print(income_stmt.columns)
    # Convert to text chunks
    text_data = "\n".join([income_stmt.to_string(), balance_sheet.to_string(), cash_flow.to_string()])
    chunks = convert_to_chunks(text_data)

    # Embed using a pre-trained model
    embeddings = embed_text(chunks, model)

    # Store in FAISS vector database
    index = store_embeddings(embeddings)

    # Initialize BM25
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    query = st.text_input("Enter your query:")
    if query:
        if not validate_query(query):
            st.error("Invalid or harmful query detected. Please enter a valid financial query.")
        else:
            retrieved_results = retrieve_similar(query, model, index, bm25, chunks)
            reranked_results, scores = rerank_results(query, retrieved_results, cross_encoder)
            filtered_results = filter_misleading_responses(list(zip(reranked_results, scores)))

            st.subheader("Top Answers:")
            if not filtered_results:
                st.write("No high-confidence answers found.")
            else:
                for i, (result, score) in enumerate(filtered_results):
                    st.write(f"**Answer {i+1}:** {result}")
                    st.write(f"Confidence Score: {score:.4f}")
