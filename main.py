from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from lavague.action_engine import ActionEngine
from lavague.defaults import DefaultLLM, DefaultEmbedder

import os
from llama_index.llms.azure_openai import AzureOpenAI

from lavague.prompts import DEFAULT_PROMPT

api_key=os.getenv("AZURE_OPENAI_KEY")
api_version="2023-05-15"
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model = "gpt-4"
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-turbo")

class LLM(AzureOpenAI):
    def __init__(self):
        super().__init__(
            model=model,
            deployment_name=deployment_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=0.0
        )
llm = LLM()
embedder = DefaultEmbedder()

action_engine = ActionEngine(llm, embedder, streaming=False)

app = FastAPI()

class InputData(BaseModel):
    query: str
    HTML: str

class OutputData(BaseModel):
    code: str
    retrieved_nodes: List[str]

@app.post("/process", response_model=OutputData)
async def process(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query = input_data.query
    html = input_data.HTML
    
    code, retrieved_nodes = action_engine.get_action(query, html)

    return OutputData(code=code, retrieved_nodes=retrieved_nodes)

@app.post("/process_direct", response_model=str)
async def process_direct(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query = input_data.query
    html = input_data.HTML
    
    prompt = DEFAULT_PROMPT.format(context_str=html, query_str=query)
    response = llm.complete(prompt)
    code = response.text
    return code