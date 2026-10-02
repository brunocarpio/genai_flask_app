from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import message_to_dict
from langchain_ibm import ChatWatsonx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from pydantic import SecretStr

from genai_flask_app.config import (
    CREDENTIALS,
    GRANITE_MODEL_ID,
    LLAMA_MODEL_ID,
    MISTRAL_MODEL_ID,
    PARAMETERS,
)


class WModel:
    def __init__(self):
        self._system = ""
        self._shared_checkpointer = InMemorySaver()
        self._thread_config = RunnableConfig(configurable={"thread_id": "1"})
        self._llama_agent = None
        self._granite_agent = None
        self._mistral_agent = None

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        self._system = system

    def __initialize_agent(self, model_id: str) -> CompiledStateGraph:
        model = ChatWatsonx(
            model_id=model_id,
            url=SecretStr(CREDENTIALS["url"]),
            project_id=CREDENTIALS["project_id"],
            api_key=SecretStr(CREDENTIALS["api_key"]),
            params=PARAMETERS,
        )
        agent = create_agent(
            model=model,
            checkpointer=self._shared_checkpointer,
            system_prompt=self._system,
        )
        return agent

    def __get_ai_response(self, agent: CompiledStateGraph, prompt: str) -> str:
        request_timestamp = datetime.now().isoformat()

        try:
            result = agent.invoke(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "additional_kwargs": {"timestamp": request_timestamp},
                        }
                    ]
                },
                config=self._thread_config,
            )

            ai_response_timestamp = datetime.now().isoformat()

            state = agent.get_state(self._thread_config)
            if state.values.get("messages"):
                last_message = state.values["messages"][-1]
                if last_message.type == "ai":
                    last_message.additional_kwargs["timestamp"] = ai_response_timestamp
                    agent.update_state(self._thread_config, values=state.values)

            return result["messages"][-1].content
        except Exception as e:
            return f"get_ai_response error: {e!s}"

    def llama_response(self, user_prompt) -> str:
        if not self._llama_agent:
            print("setting up llama agent")
            self._llama_agent = self.__initialize_agent(LLAMA_MODEL_ID)
            print("finished setting up llama agent")
        return self.__get_ai_response(self._llama_agent, user_prompt)

    def granite_response(self, prompt) -> str:
        if not self._granite_agent:
            print("setting up granite agent")
            self._granite_agent = self.__initialize_agent(GRANITE_MODEL_ID)
            print("finished setting up granite agent")
        return self.__get_ai_response(self._granite_agent, prompt)

    def mistral_response(self, user_prompt) -> str:
        if not self._mistral_agent:
            print("setting up mistral agent")
            self._mistral_agent = self.__initialize_agent(MISTRAL_MODEL_ID)
            print("finished setting up mistral agent")
        return self.__get_ai_response(self._mistral_agent, user_prompt)

    def get_chat_history(self) -> list[dict[str, Any]]:
        active_agent = None

        if self._granite_agent:
            active_agent = self._granite_agent
        elif self._mistral_agent:
            active_agent = self._mistral_agent
        elif self._llama_agent:
            active_agent = self._llama_agent
        if not active_agent:
            return []

        messages = active_agent.get_state(self._thread_config).values.get(
            "messages", []
        )
        messages_dict = [message_to_dict(message) for message in messages]

        return messages_dict

    def __str__(self) -> str:
        return f"WModel(system={self._system})"
