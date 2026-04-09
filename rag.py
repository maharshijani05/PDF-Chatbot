from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config import VECTORSTORE_DIR, GOOGLE_API_KEY
from ingest import load_vectorstore

SYSTEM_PROMPT = PromptTemplate(
    template="""
You are a strict retrieval-based QA system.

Your job is to answer the question using ONLY the provided context.
Do NOT use prior knowledge.

====================
RULES (MANDATORY)
====================

1. SOURCE OF TRUTH
- Use ONLY the given context.
- If the answer is not explicitly present, respond: "I don't know".

2. COMPLETENESS
- If multiple parts of the context contribute to the answer, include ALL of them.
- Do NOT give partial answers.

3. NO HALLUCINATION
- Do NOT infer, assume, or fabricate information.
- If context is ambiguous or insufficient → "I don't know".

4. CONFLICT RESOLUTION
- If context contains conflicting information:
  - Report both versions.
  - Do NOT choose arbitrarily.

5. ANSWER STRUCTURE
- Be clear and structured.
- Use bullet points for multiple facts.
- Keep it concise but complete.

6. RELEVANCE FILTERING
- Ignore irrelevant parts of the context.
- Focus only on information needed to answer the question.

7. NO EXTRA TEXT
- Do NOT add explanations about your process.
- Do NOT restate the question.

8. REQUIRED OUTPUT FORMAT
- Use the exact structure:
  Answer: <your answer>
  Source: <context reference>
- If the answer cannot be found in the context, reply only with:
  Answer: I don't know
  Source: N/A

====================
CONTEXT
====================
{context}

====================
QUESTION
====================
{question}

====================
ANSWER
====================
""",
    input_variables=["context", "question"],
)


_retriever = None


def _build_retriever(vectorstore_path: Path = VECTORSTORE_DIR):
    global _retriever
    if _retriever is None:
        _retriever = load_vectorstore(vectorstore_path).as_retriever(search_type="mmr", search_kwargs={"k": 6})
    return _retriever


def invoke_with_context(question: str) -> tuple[str, list[str]]:
    """Run the RAG chain and return (answer, contexts). Used when raw contexts are needed."""
    retriever = _build_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    answer = (SYSTEM_PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, [doc.page_content for doc in docs]


def build_qa_chain(vectorstore_path: Path = VECTORSTORE_DIR):
    """Build a reusable LangChain pipeline."""
    retriever = _build_retriever(vectorstore_path)
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | SYSTEM_PROMPT
        | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        | StrOutputParser()
    )
