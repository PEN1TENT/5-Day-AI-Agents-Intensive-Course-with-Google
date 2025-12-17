# %%
from copy import PyStringMap
import os

from dotenv import load_dotenv
from pydantic.types import NonNegativeFloat
from pygments.token import Number
from requests.sessions import session
from sqlalchemy.sql.coercions import TruncatedLabelImpl

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("Gemini API key setup complete.")

# %%ฃ
import logging 
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Cleaned up {log_file}")
        
# Configure logging with DEBUG log level.
logging.basicConfig(
    filename = "logger.log",
    level = logging.DEBUG,
    format = "%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("Logging configured")

# %%
from IPython.core.display import display, HTML

def get_adk_proxy_url(port="8000", host="localhost"):
    """
    Generates a link for the ADK Web UI running on a local Jupyter environment.
    """
    # In a local environment, the URL is direct
    url = f"http://{host}:{port}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #2ecc71; border-radius: 8px; background-color: #f4fff8; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>✅ Local Environment Detected</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI will be accessible locally. 
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (<code>!adk web ...</code>) to start the service.</li>
                <li>Once the cell says it is running, click the button below.</li>
            </ol>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #2ecc71; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            Open Local ADK Web UI ↗
        </a>
    </div>
    """

    display(HTML(styled_html))
    return url

print("Local helper function defined.")

# %%
## Hands-On Debuggin with ADK Web UI
!adk create research-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

# %%
url_prefix = get_adk_proxy_url()

# %%
!adk web --log_level DEBUG --url_prefix {url_prefix}

# %%
# Check the DEBUG logs form the broken agent
print("Examining web server logs for debugging clues...\n")
!cat logger.log

# %%
## Logging in Production
print("----- EXAMPLE PLUGIN - DOES NOTHING -----")

import logging
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin

# Applies to all agent and model cells
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""
    
    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name = "count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0
        
    # Callback 1: Runs before an agent is called. You can add any custom logic here.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext) -> None:
        """Count agent runs."""
        self.agent_cont += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")
        
    # Callback 2: Runs before a model is called. You can add any custom logic here.
    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest)-> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")

# %%
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts = 5, # Maximum retry attempts
    exp_base = 7, # Delay multiplier
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504], # Retry on these HTTP errors
)

def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Arg: 
        papers: A list of strings, where each string is a research paper.
    Return: 
        The number of papers in the list.
    """
    return len(papers)
    
# Google search agent
google_search_agent = LlmAgent(
    name = "google_search_agent",
    model = Gemini(model = "gemini-2.5-flash-lite", retry_options = retry_config),
    description = "Searches for information using Google search",
    instruction = "Use the google_search tool to find information on the given topic. Return the raw search results.",
    tools = [google_search],
)

# Root agent
research_agent_with_plugin = LlmAgent(
    name = "research_paper_finder_agent",
    model = Gemini(model = "gemini-2.5-flash-lite", retry_options = retry_config),
    instruction = """Your task is to find research papers and count them.
    You must follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'.
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.
    """,
    tools = [AgentTool(agent = google_search_agent), count_papers],
)

print("Agent created")

# %%
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (LoggingPlugin,) # <---- 1. Import the Plugin
from google.genai import types
import asyncio

runner = InMemoryRunner(
    agent = research_agent_with_plugin,
    plugins = [
        LoggingPlugin()
    ], # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("Runner configured")

# %%
print("Running agent with LoggingPlugin...")
print("Watch the comprehensive logging output below:\n")

reponse = await runner.run_debug("Find recent papers on quatum computing")
