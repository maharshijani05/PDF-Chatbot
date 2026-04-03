from pathlib import Path
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from config import VECTORSTORE_DIR, GOOGLE_API_KEY
from ingest import load_vectorstore

RAG_PROMPT = PromptTemplate(
    template="""You are a helpful assistant.

Answer the question ONLY using the context below.

IMPORTANT:
- If the context contains multiple steps or methods, include ALL of them.
- Do NOT give partial answers.
- Structure the answer clearly in points.
- Ensure completeness over brevity.

If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def rerank_docs(query: str, docs: List, top_k: int = 4) -> List:
    query_emb = embeddings.embed_query(query)

    scored = []
    for doc in docs:
        doc_emb = embeddings.embed_query(doc.page_content)
        score = dot_product(query_emb, doc_emb)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

def build_qa_chain(vectorstore_path: Path = VECTORSTORE_DIR):
    vectorstore = load_vectorstore(vectorstore_path)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    def build_context(question: str) -> str:
        docs = retriever.invoke(question)
        docs = rerank_docs(question, docs)
        return "\n\n".join(d.page_content for d in docs)

    return (
        {
            "context": build_context,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )