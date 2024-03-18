from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from .default_prompt import DEFAULT_PROMPT

MAX_CHARS = 1500
K = 3

import re

def extract_first_python_code(markdown_text):
    # Pattern to match the first ```python ``` code block
    pattern = r"```python(.*?)```"
    
    # Using re.DOTALL to make '.' match also newlines
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        # Return the first matched group, which is the code inside the ```python ```
        return match.group(1).strip()
    else:
        # Return None if no match is found
        return None

class ActionEngine:
    def __init__(self, llm, embedding,
                 prompt_template=None, cleaning_function=None):

        self.llm = llm
        self.embedding = embedding
        if not prompt_template:
            prompt_template = DEFAULT_PROMPT
            
        if not cleaning_function:
            cleaning_function = extract_first_python_code
            
        self.prompt_template = prompt_template
        self.cleaning_function = cleaning_function

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

    # WARNING: streaming is broken for now
    def get_query_engine(self, state):
        html = state
        index = self._get_index(html)

        retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=K,
        )

        # No streaming for now
        response_synthesizer = get_response_synthesizer(streaming=False, llm=self.llm)

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        prompt_template = PromptTemplate(self.prompt_template)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": prompt_template}
        )

        return query_engine

    def get_action(self, query, html):
        query_engine = self.get_query_engine(html)
        response = query_engine.query(query)
        source_nodes = response.get_formatted_sources(MAX_CHARS)
        code = response.response
        code = self.cleaning_function(code)
        return code, source_nodes