from genai_flask_app.config import (  # noqa: I001
    CREDENTIALS,
    GRANITE_MODEL_ID,
    LLAMA_MODEL_ID,
    MISTRAL_MODEL_ID,
    PARAMETERS,
)
from langchain.agents import create_agent
from langchain_ibm import ChatWatsonx
from langgraph.checkpoint.memory import InMemorySaver


class AppState:
    system_prompt = ""

    __llama_agent = None
    __granite_agent = None
    __mistral_agent = None

    def __init__(self, system_prompt):
        if not system_prompt:
            raise ValueError("system_prompt cannot be empty or null")
        self.system_prompt = system_prompt

    def __initialize_model(self, model_id):
        model = ChatWatsonx(
            model_id=model_id,
            url=CREDENTIALS["url"],
            project_id=CREDENTIALS["project_id"],
            api_key=CREDENTIALS["api_key"],
            params=PARAMETERS,
        )
        return model

    def __initialize_agent(self, model_id):
        model = self.__initialize_model(model_id)
        agent = create_agent(
            model=model, checkpointer=InMemorySaver(), system_prompt=self.system_prompt
        )
        return agent

    def __get_ai_response(self, agent, user_prompt):
        thread_config = {"configurable": {"thread_id": "1"}}
        ai_response = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            thread_config,
        )
        return ai_response["messages"][-1].content

    def llama_response(self, user_prompt):
        if not self.__llama_agent:
            print("setting up llama agent")
            self.__llama_agent = self.__initialize_agent(LLAMA_MODEL_ID)
            print("finished setting up llama agent")
        return self.__get_ai_response(self.__llama_agent, user_prompt)

    def granite_response(self, user_prompt):
        if not self.__granite_agent:
            self.__granite_agent = self.__initialize_agent(GRANITE_MODEL_ID)
        return self.__get_ai_response(self.__granite_agent, user_prompt)

    def mistral_response(self, user_prompt):
        if not self.__mistral_agent:
            self.__mistral_agent = self.__initialize_agent(MISTRAL_MODEL_ID)
        return self.__get_ai_response(self.__mistral_agent, user_prompt)
