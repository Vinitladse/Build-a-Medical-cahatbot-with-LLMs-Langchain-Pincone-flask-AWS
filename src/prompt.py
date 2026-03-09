from langchain_core.prompts import ChatPromptTemplate

system_prompt = ChatPromptTemplate.from_template(
    """
You are a medical assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)
