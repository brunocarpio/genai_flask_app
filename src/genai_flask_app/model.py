from config import (
    CREDENTIALS,
    GRANITE_MODEL_ID,
    LLAMA_MODEL_ID,
    MISTRAL_MODEL_ID,
    PARAMETERS,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx
from pydantic import BaseModel, Field


class AIResponse(BaseModel):
    summary: str = Field(description="summary of the user's message")
    sentiment: str = Field(
        description="sentiment score from 0 (negative) to 100 (positive)"
    )
    response: str = Field(
        description="suggested response to the user"
    )


def initialize_model(model_id):
    model = ChatWatsonx(
        model_id=model_id,
        url=CREDENTIALS["url"],
        project_id=CREDENTIALS["project_id"],
        api_key=CREDENTIALS["api_key"],
        params=PARAMETERS,
    )
    return model.with_structured_output(AIResponse)


llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)


template = ChatPromptTemplate(
    [
        ("system", "{system_prompt}"),
        ("human", "{user_prompt}"),
    ]
)


def get_ai_response(model, system_prompt, user_prompt):
    chain = template | model
    return chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    })


def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, system_prompt, user_prompt)


def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, system_prompt, user_prompt)


def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, system_prompt, user_prompt)
