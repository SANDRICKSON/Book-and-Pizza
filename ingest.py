import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document

load_dotenv()

PINECONE_INDEX_NAME = "langchain-books-pure-v1"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"🔧 Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created.")
else:
    print("ℹ️ Index already exists.")

print("🚀 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

print("📄 Creating dummy documents...")
docs = [
    Document(
        page_content="A thrilling dystopian novel about survival and rebellion.",
        metadata={"title": "Hunger Games", "authors": "Suzanne Collins"}
    ),
    Document(
        page_content="A magical school where a boy discovers his destiny.",
        metadata={"title": "Harry Potter", "authors": "J.K. Rowling"}
    ),
    Document(
        page_content="A journey through space and time to save humanity.",
        metadata={"title": "Interstellar Dreams", "authors": "Unknown"}
    )
]

print("🧠 Adding to vectorstore...")
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("✅ Ingestion complete.")