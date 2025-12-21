# %%
import random
import time
from google.cloud.aiplatform_v1.services.gen_ai_tuning_service.transports.rest_base import json
import vertexai
from vertexai import agent_engines

print("Imports completed successfully")

# %%
## Set your PROJECT_ID
import os
PROJECT_ID = "your-project-id" # TODO: Replace with your project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-project-id" or not PROJECT_ID:
    raise ValueError("Please replace 'your-project-id' with your actual Google Cloud Project ID.")
    
print(f"Project ID set to: {PROJECT_ID}")

# %%
## Create simple agent - all code for the agent will live in this directory
!mkdir -p sample_agent

print(f"Sample Agent directory created")

# %%
regions_list = ["europe-west", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)

print(f"Selected deployment region: {deployed_region}")

# %%
!adk deploy agent_engine --project = $PROJECT_ID --region = $deployed_region sample_agent --agent_engine_config_file = sample_agent/.agent_engine_config.json

# %%
# Initialize Vertex AI
vertexai.init(project = PROJECT_ID, location = deployed_region)

# Get the most recontly deployed agent
agents_list = list(agent_engines.list())
if agents_list: 
    remote_agent = agents_list[0] # Get the first (most recent) agent
    client = agent_engines
    print(f"Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("No agents found. Please deploy first.")
    
# %%
async for item in remote_agent.async_stream_query(
    message = "What is the weather in Tokyou?",
    user_id = "user_42",
):
    print(item)
    
# %%
agent_engines.delete(resource_name = remote_agent.resource_name, force = True)