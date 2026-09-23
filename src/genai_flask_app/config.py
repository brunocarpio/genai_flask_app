import os

from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")

if not api_key:
    raise ValueError("critical error: WATSONX_API_KEY missing or empty")

if not project_id:
    raise ValueError("critical error: WATSONX_PROJECT_ID missing or empty")


PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 128,
}

CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "api_key": api_key,
    "project_id": project_id,
}

LLAMA_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
