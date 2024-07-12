# import locale

import gradio as gr
from llama_index.core import Document
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core import get_response_synthesizer
# Monkey patch because stream_complete is not implemented in the current version of llama_index
from llama_index.core.base.llms.types import (
    CompletionResponse,
)
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.retrievers.bm25 import BM25Retriever
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# from action_engine import ActionEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# locale.getpreferredencoding = lambda: "UTF-8"

## chatglm modle
llm = Ollama(model='llama3', request_timeout=2000, )
# embed_model = "llama3"
embedder = OllamaEmbedding(model_name="llama3", )

with open("prompt_template.txt", "r") as file:
    PROMPT_TEMPLATE_STR = file.read()

# Preparing the action engine


MAX_CHARS = 1500
K = 3


class ActionEngine:
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding

    def _get_index(self, html):
        text_list = [html]
        documents = [Document(text=t) for t in text_list]

        splitter = CodeSplitter(
            language="html",
            chunk_lines=40,  # lines per chunk
            chunk_lines_overlap=200,  # lines overlap between chunks
            max_chars=MAX_CHARS,  # max chars per chunk
        )
        nodes = splitter.get_nodes_from_documents(documents)
        nodes = [node for node in nodes if node.text]

        index = VectorStoreIndex(nodes, embed_model=self.embedding)

        return index

    def get_query_engine(self, state):
        html = state
        index = self._get_index(html)

        retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=K,
        )

        response_synthesizer = get_response_synthesizer(streaming=True, llm=self.llm)

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        prompt_template = PromptTemplate(PROMPT_TEMPLATE_STR)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": prompt_template}
        )

        return query_engine


# Code execution in action
action_engine = ActionEngine(llm, embedder)

## Setup chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=1600,900")
# webdriver_service = Service("chromedriver")
# Set path to chrome/chromedriver as per your configuration

# try:
# import google.colab
# chrome_options.binary_location = "/Users/crisschan/workspace/PySpace/laVague/content/chrome-linux64/chrome"
webdriver_service = Service("{driverHome}\\chromedriver.exe")
# except:
#     import os.path
#     homedir = os.path.expanduser("~")
#     chrome_options.binary_location = f"{homedir}/chrome-linux64/chrome"
#     webdriver_service = Service(f"{homedir}/chromedriver-linux64/chromedriver")

# Choose Chrome Browser
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

title = """
<div align="center">
  <h1> Welcome to Ollama3-LaVague</h1>
  <p>Redefining internet surfing by transforming natural language instructions into seamless browser interactions.</p>
</div>
"""


# action_engine = ActionEngine(llm, embedder)

def process_url(url):
    driver.get(url)
    driver.save_screenshot("screenshot.png")
    # This function is supposed to fetch and return the image from the URL.
    # Placeholder function: replace with actual image fetching logic.
    return "screenshot.png"


def process_instruction(query):
    state = driver.page_source
    query_engine = action_engine.get_query_engine(state)
    streaming_response = query_engine.query(query)

    source_nodes = streaming_response.get_formatted_sources(MAX_CHARS)

    response = ""

    for text in streaming_response.response_gen:
        # do something with text as they arrive.
        print(text)
        response += text
        yield response, source_nodes


def exec_code(code):
    code = "#" + code.split("```")[1]
    # print(code)
    try:
        # if len(code)==0:
        #     return  "No code Generated"
        # else:
        exec(code)
        print(code)
        return "Successful code execution", code
    except Exception as e:
        output = f"Error in code execution: {str(e)}"
        return output, code


def update_image_display(img):
    driver.save_screenshot("screenshot.png")
    url = driver.current_url
    return "screenshot.png", url


def create_demo(base_url, instructions):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.HTML(title)
        with gr.Row():
            url_input = gr.Textbox(value=base_url, label="Enter URL and press 'Enter' to load the page.")

        with gr.Row():
            with gr.Column(scale=8):
                image_display = gr.Image(label="Browser", interactive=False)

            with gr.Column(scale=2):
                text_area = gr.Textbox(label="Instructions")
                gr.Examples(examples=instructions, inputs=text_area,

                            )
                generate_btn = gr.Button(value="Execute")
                code_display = gr.Code(label="Generated code", language="python",
                                       lines=5, interactive=False)
                with gr.Accordion(label="Logs", open=False) as log_accordion:
                    log_display = gr.Textbox(interactive=False)
                    source_display = gr.Textbox(label="Retrieved nodes", interactive=False)
        # Linking components
        url_input.submit(process_url, inputs=url_input, outputs=image_display)
        generate_btn.click(process_instruction, inputs=text_area, outputs=[code_display, source_display]).then(
            exec_code, inputs=code_display, outputs=[log_display, code_display]
        ).then(
            update_image_display, inputs=image_display, outputs=[image_display, url_input]
        )
    demo.launch(share=False)


if __name__ == "__main__":
    base_url = "http://127.0.0.1:7860/"

    instructions = [
        "Enter test1234@google.com in the first input box，Then enter test1234 in the second input box, and then click SIGN IN button，Wait for the new page to load",
        "Click the APPS icon under Apps on the page, Wait for the new page to load",
        "Wait for the page element Add New to appear，then click Add New button",
        "Click Experiment that under Add New, Wait for the new page to load"]

    create_demo(base_url, instructions)