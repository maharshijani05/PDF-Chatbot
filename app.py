import os

import streamlit as st

from config import RAW_DATA_DIR, VECTORSTORE_DIR, bootstrap_environment
from ingest import create_vectorstore, load_pdf, split_documents
from rag import build_qa_chain

bootstrap_environment()

st.set_page_config(page_title="PDF Q&A Chatbot")
st.title("📄 PDF Q&A Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing environment variable: `GOOGLE_API_KEY`")
    st.info("Add the key in your `.env` file, then restart the app.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if not uploaded_file:
    st.info("Upload a PDF to start chatting with your document.")
    st.stop()

file_path = RAW_DATA_DIR / uploaded_file.name
file_path.write_bytes(uploaded_file.read())
st.success("PDF uploaded successfully.")


@st.cache_resource
def process_pdf(path: str):
    return create_vectorstore(split_documents(load_pdf(path)), save_path=VECTORSTORE_DIR)


@st.cache_resource
def get_chain():
    return build_qa_chain(vectorstore_path=VECTORSTORE_DIR)


try:
    process_pdf(str(file_path))
    qa_chain = get_chain()
except Exception as exc:
    st.error("Failed to process the PDF or initialize the chain.")
    st.exception(exc)
    st.stop()

st.success("PDF processed and indexed.")

query = st.text_input("Ask a question about the PDF:")
if not query:
    st.stop()

with st.spinner("Thinking..."):
    try:
        answer = qa_chain.invoke(query)
    except Exception as exc:
        st.error("Could not generate an answer.")
        st.exception(exc)
        st.stop()

st.markdown("### Answer")
st.write(answer)
