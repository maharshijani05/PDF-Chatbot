from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import VECTORSTORE_DIR, GOOGLE_API_KEY

EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_pdf(file_path: str) -> list[Document]:
    return PyPDFLoader(file_path).load()


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_documents(documents)


def create_vectorstore(docs: list[Document], save_path: Path = VECTORSTORE_DIR) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    return vectorstore


def load_vectorstore(load_path: Path = VECTORSTORE_DIR) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    return FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
