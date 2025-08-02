# ingest.py
import os
import requests
import pypdf
import io
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.utils import enhanced_text_cleaning

load_dotenv()

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hackathon-policy-index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def run_ingestion():
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    print(f"Starting ingestion for: {document_url}")

    # 1. Download, clean, and chunk text
    response = requests.get(document_url)
    reader = pypdf.PdfReader(io.BytesIO(response.content))
    cleaned_text = enhanced_text_cleaning("".join(p.extract_text() for p in reader.pages if p.extract_text()))
    chunks = [cleaned_text[i:i + 1200] for i in range(0, len(cleaned_text), 1000)]
    print(f"Document processed into {len(chunks)} chunks.")

    # 2. Initialize Modern Pinecone Client and Services
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # 3. Connect to the index
    index = pc.Index(INDEX_NAME)

    # 4. Embed and Upload
    print("Uploading to Pinecone...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeds = embeddings.embed_documents(batch)
        index.upsert(vectors=zip(ids, embeds, [{"text": text} for text in batch]))
        print(f"Upserted batch {i//batch_size + 1}")

    print(f"✅ Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()