from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from lavague.core import WorldModel, ActionEngine
from lavague.core.agents import WebAgent
from lavague.core.context import Context
from lavague.drivers.selenium import SeleniumDriver


# Customize the LLM, multi-modal LLM and embedding models
llm = Gemini(model_name="models/gemini-1.5-flash-latest")
mm_llm =  OpenAIMultiModal(model="gpt-4o", temperature=0.0)
embedding = GeminiEmbedding(model_name="models/text-embedding-004")

# Initialize context with our custom elements
context = Context(llm, mm_llm, embedding)

# Initialize the Selenium driver
selenium_driver = SeleniumDriver()

# Initialize a WorldModel and ActionEnginem passing them the custom context
world_model = WorldModel.from_context(context)
action_engine = ActionEngine.from_context(context, selenium_driver)

# Create your agent
agent = WebAgent(world_model, action_engine)

agent.get("https://huggingface.co/docs")
agent.run("Go on the quicktour of PEFT")