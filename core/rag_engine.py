import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough

from core.vector_store import build_vector_store,load_vector_store,get_retriever


def get_llm():
  return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)


def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript) :
  vector_store = build_vector_store(transcript)
  retriever = get_retriever(vector_store,k= 4)
  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI meeting assistant.

Answer the user's question using ONLY the provided meeting context.

If the answer is not present in the context, say:
"I couldn't find that information in the meeting transcript."

Context:
{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )
  rag_chain =(
    {"context":retriever| RunnableLambda(format_docs),
     "question": RunnablePassthrough()} | prompt |llm | StrOutputParser()
  )
  return rag_chain


def load_rag_chain():
  vector_store = load_vector_store()
  retriever = get_retriever(vector_store)
  llm = get_llm()
  prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI meeting assistant.

Answer the user's question using ONLY the provided meeting context.

If the answer is not present in the context, say:
"I couldn't find that information in the meeting transcript."

Context:
{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )
  rag_chain =(
    {"context":retriever| RunnableLambda(format_docs),
     "question": RunnablePassthrough()} | prompt |llm | StrOutputParser()
  )
  return rag_chain
  
def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)