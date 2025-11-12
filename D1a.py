# %%
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Gemini API key setup complete.")

# %%
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("ADK components imported successfully.")
# %%
# Define helper funcitons that will be reused throughout the not ebook

import webbrowser

try:
    from IPython.display import display, HTML
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False

def get_local_service_url(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> str:
    #    !adk web --host 127.0.0.1 --port 8000
    url = f"http://{host}:{port}"

    if _HAS_IPY:
        styled_html = f"""
        <div style="padding: 15px; border: 2px solid #3b82f6; border-radius: 8px; background-color: #f5f9ff; margin: 20px 0;">
            <div style="font-family: sans-serif; margin-bottom: 12px; color: #0f172a; font-size: 1.05em;">
                <strong>Local Web UI</strong>
            </div>
            <div style="font-family: sans-serif; margin-bottom: 15px; color: #0f172a; line-height: 1.5;">
                Make sure your app is running locally on <code>{host}:{port}</code> (e.g. <code>adk web --host {host} --port {port}</code>),
                then click the button below.
            </div>
            <a href="{url}" target="_blank" style="
                display: inline-block; background-color: #2563eb; color: white; padding: 10px 20px;
                text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 600;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
                Open Local Web UI ↗
            </a>
        </div>
        """
        display(HTML(styled_html))
    else:
        print(f"Local Web UI: {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return url


print("Local helper function defined.")

# %%
retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes=[429, 500, 503, 504]
)

# %%
#https://google.github.io/adk-docs/agents/

root_agent = Agent(
    name = "helpful_assistant",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    description = "A simple agent that can answer general question.",
    instruction = "You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools = [google_search]
)

print("Root Agent defined.")

# %%
runner = InMemoryRunner(agent = root_agent)

print("Runner created.")
#https://google.github.io/adk-docs/runtime/#execution-logics-role-agent-tool-callback
# %%
reponse = await runner.run_debug(
    "What is Agent Development Kit from Google? What languages is the SDK available in?"
)

#%%
response = await runner.run_debug("What's the weather in London?")

# %%
url_prefix = get_local_service_url(open_browser = True)

# %%
!adk web --url_prefix {url_prefix}

# %%
