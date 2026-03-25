import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOC_PATH = "docs/master_policy_doc.txt"
DB_DIR = "chroma_db"

def ingest_data():
    if os.path.exists(DB_DIR):
        print("Clearing old database...")
        shutil.rmtree(DB_DIR)

    print("Loading master document...")
    loader = TextLoader(DOC_PATH)
    documents = loader.load()

    print("Chunking text with larger chunk limits...")
    text_splitter = RecursiveCharacterTextSplitter(
        # INCREASED CHUNK SIZE: Fits whole product sections
        chunk_size=1200, 
        # INCREASED OVERLAP: Ensures no bullet points get orphaned
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} conceptual chunks.")

    print("Generating Embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    print("✅ Ingestion complete! Database is ready.")

if __name__ == "__main__":
    ingest_data()