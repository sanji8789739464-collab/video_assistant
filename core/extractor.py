import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ==========================================================
# LLM
# ==========================================================

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


# ==========================================================
# Generic Chain Builder
# ==========================================================

def build_chain(system_prompt: str):
    """
    Builds:
    Input -> Prompt -> LLM -> String Output
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


# ==========================================================
# Action Items
# ==========================================================

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting assistant.

Extract every actionable task from the meeting transcript.

For every action item include:
- Task
- Owner (if mentioned, otherwise "Unassigned")
- Deadline (if mentioned, otherwise "Not specified")

Return the output as bullet points.
"""
    )

    return chain.invoke(transcript)


# ==========================================================
# Questions
# ==========================================================

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting assistant.

Extract every question asked during the meeting.

Rules:
- Return only the questions.
- Preserve the original wording as much as possible.
- Remove duplicate questions.
- Present each question as a bullet point.
- If no questions were asked, return:
"No questions found."
"""
    )

    return chain.invoke(transcript)


# ==========================================================
# Key Decisions
# ==========================================================

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting assistant.

Extract every important decision made during the meeting.

Rules:
- Include only finalized decisions.
- Ignore brainstorming and open discussions.
- Present each decision as a bullet point.
- If no decisions were made, return:
"No key decisions were made."
"""
    )

    return chain.invoke(transcript)