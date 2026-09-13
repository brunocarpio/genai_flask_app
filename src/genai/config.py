from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from dotenv import load_dotenv
import os

load_dotenv()

PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "api_key": os.environ.get("WATSONX_APIKEY", ""),
    "project_id": os.environ.get("WATSONX_PROJECT_ID", "")
}

LLAMA_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
