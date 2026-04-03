from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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


def build_qa_chain(vectorstore_path: Path = VECTORSTORE_DIR):
    retriever = load_vectorstore(vectorstore_path).as_retriever(search_type="mmr",search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
