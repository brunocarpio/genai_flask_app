from model import AppState


def call_all_models(user_prompt):
    llama_result = appState.llama_response(user_prompt)
    granite_result = appState.granite_response(user_prompt)
    mistral_result = appState.mistral_response(user_prompt)

    print("llama response:\n", llama_result)
    print("\ngranite response:\n", granite_result)
    print("\nmistral response:\n", mistral_result)


appState = AppState(
    "You are a helpful assistant who provides concise and accurate answers"
)


call_all_models("What is the capital of Canada?")

print("\n----------follow up questions----------\n")

call_all_models("Tell me a fun fact about it")
