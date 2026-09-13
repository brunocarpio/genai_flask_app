from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from dotenv import load_dotenv
import os

load_dotenv()


def main() -> None:
    credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=os.environ.get("WATSONX_APIKEY", "")
            )

    params = {
            GenTextParamsMetaNames.DECODING_METHOD: "greedy",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
            }

    model = ModelInference(
            model_id="ibm/granite-4-h-small",
            # model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            # model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
            params=params,
            credentials=credentials,
            project_id=os.environ.get("WATSONX_PROJECT_ID", "")
            )

    # text = """
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # You are an expert assistant who provides concise and accurate answers.<|eot_id|>
    #
    # <|start_header_id|>user<|end_header_id|>
    # What is the capital of Canada?<|eot_id|>
    #
    # <|start_header_id|>assistant<|end_header_id|>
    # """

    text = """
    <|system|>You are an expert assistant who provides concise and accurate answers.
    <|user|>What is the capital of Canada?
    <|assistant|>
    """

    print(model.generate(text)['results'][0]['generated_text'])


if __name__ == "__main__":
    main()
