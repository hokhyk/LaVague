from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
import os

def get_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
    import os.path
    
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1600,900")
    
    homedir = os.path.expanduser("~")
    chrome_options.binary_location = f"{homedir}/chrome-linux64/chrome"
    webdriver_service = Service(f"{homedir}/chromedriver-linux64/chromedriver")
    
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    return driver

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_LLM = "HuggingFaceH4/zephyr-7b-gemma-v0.1"
DEFAULT_LLM = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
DEFAULT_MAX_NEW_TOKENS = 512
HF_TOKEN = os.environ["HF_TOKEN"]

class DefaultEmbedder(HuggingFaceEmbedding):
	def __init__(self, model_name=DEFAULT_EMBED_MODEL, device="cuda"):
		super().__init__(model_name, device)

class DefaultLLM(HuggingFaceInferenceAPI):
	def __init__(self, model_name = DEFAULT_LLM, token=HF_TOKEN, num_output=DEFAULT_MAX_NEW_TOKENS):
		super().__init__(model_name=model_name, token=token, num_output=num_output)
