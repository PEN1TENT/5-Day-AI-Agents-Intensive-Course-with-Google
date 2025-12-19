# %%
import os

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("Gemini API key setup complete.")

# %%
import json 
import requests
import subprocess
import time
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("ADK components imported successfully.")

# %%
retry_config = types.HttpRetryOptions(
    attempts = 5, # Maximum retry attempts
    exp_base = 7, # Delay multiplier
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504], # Retry on these HTTP errors
)

## %%
## Create the Product Catalog Agent (To Be Exposed)
# Define aproduct catalog lookup tool
# In a real system, this would query the vendor's product database
def get_product_info(product_name: str) -> str:
    """Get product information for a given product.
    Args:
        product_name: Name of the product (e.g., "iPhone 15 Pro", "MacBook Pro")
    Return:
        Product information as a string
    """
    # Mock product catalog - in production, this would query a real database
    product_catalog = {
        "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
        "sumsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
        "dell xps 15": 'Dell XPS 15, $1,299, In Stock (45 units), 15.6" display, 16GB RAM, 512GB SSD',
        "macbook pro 14": 'MacBook Pro 14", $1,999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD',
        "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
        "ipad air": 'iPad Air, $599, In Stock (28 units), 10.9" display, 64GB',
        "lg ultrawide 34": 'LG UltraWide 34" Monitor, $499, Out of Stock, Expected: Next week',
    }
    
    product_lower = product_name.lower().strip()
    
    if product_lower in product_catalog:
        return f"Product: {product_catalog[product_lower]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, I don't have information for {product_name}. Available products: {available}"
        
# Create the Product Catalog Agent
# This agent specializes in providing product information from the vendor's catalog
product_catalog_agent = LlmAgent(
    model = Gemini(model = "gemini-2.5-flash-lite", retry_options = retry_config),
    name = "product_catalog_agent",
    description = "External vendor's product catalog agent that provides product information and availability.",
    instruction = """
    You are a product catalog specialist from an external vendor.
    When asked about products, use the get_product_info tool to fetch data from the catalog.
    Provide clear, accurate product information including price, availability, and specs.
    If asked about multiple products, look up each one.
    Be professional and helpful.
    """,
    tools = [get_product_info], # Register the product lookup tool
)

print("Product Catalog Agent created successfully!")
print("   Model: gemini-2.5-flash-lite")
print("   Tool: get_product_info()")
print("   Ready to be exposed via A2A...")

# %%
## Expose the Product Catalog Agent via A2A
# Convert the product catalog agent to an A2A-compatible application
# This creates a FastAPI/Starlette app that:
#   1. Serves the agent at the A2A protocol endpoints
#   2. Provides an auto-generated agent card
#   3. Handles A2A communication protocol
product_catalog_a2a_app = to_a2a(
    product_catalog_agent, port = 8001 # Port where this agent will be served
)

print("Product Catalog Agent is now A2A-compatible!")
print("   Agent will be served at: http//localhost:8001")
print("   Agent card will be at: http://localhost:8001/.well-known/agent-card.json")
print("   Ready to start the server...")

# %%
## Start the Product Catalog Agent Server
# First, let's save the product catalog agent to a file that uvicorn can import
product_catalog_agent_code = '''
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504],
)

def get_product_info(product_name: str) -> str:
    """Get product information for a given product."""
    product_catalog = {
        "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
        "samsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
        "dell xps 15": "Dell XPS 15, $1,299, In Stock (45 units), 15.6\\" display, 16GB RAM, 512GB SSD",
        "macbook pro 14": "MacBook Pro 14\\", $1,999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD",
        "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
        "ipad air": "iPad Air, $599, In Stock (28 units), 10.9\\" display, 64GB",
        "lg ultrawide 34": "LG UltraWide 34\\" Monitor, $499, Out of Stock, Expected: Next week",
    }
        
    product_lower = product_name.lower().strip()
    
    if product_lower in product_catalog:
        return f"Product: {product_catalog[product_lower]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, I don't have information for {product_name}. Available products: {available}"
        
product_catalog_agent = LlmAgent(
    model = Gemini(model = "gemini-2.0-flash-lite", retry_options = retry_config),
    name = "product_catalog_agent",
    description = "External vendor's product catalog agent.",
    instruction = "Provide clear, accurate product information.",
    tools = [get_product_info]
)
    
app = to_a2a(product_catalog_agent, port = 8001)
'''
# Write the product catalog agent to a temporary file
with open ("/tmp/product_catalog_server.py", "w") as f:
    f.write(product_catalog_agent_code)
    
print("Product Catalog agent code saved to /tmp/product_catalog_server.py")

os.system("pkill -f 'uvicorn product_catalog_server:app'")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "product_catalog_server:app", # Module:app format
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd = "/tmp", # Run from /tmp where the file is
    stdout = subprocess.PIPE,
    stderr = subprocess.PIPE,
    env = {**os.environ}, # Pass environment variables (including GOOGLE_API_KEY)
)

print("Starting Product Catalog Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout = 1
        )
        if response.status_code == 200:
            print(f"\nProduct Catalog Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end = "", flush = True)
else:
    print("\nServer may not be ready yet. Check manually if needed.")
    
    # Store the process so we can stop it later
    globals()["product_catalog_server_process"] = server_process

# %%
# Fetch the agent card from the running server
try: 
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout = 5
    )
    
    if response.status_code == 200:
        agent_card = response.json()
        print("Product Catalog Agent Card:")
        print(json.dumps(agent_card, indent = 2))
        
        print("\nKey Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capavilities exposed")
    else:
        print(f"Failed to fetch agent card: {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"Error fetching agent card: {e}")
    print("   Make sure the Product Catalog Agent server is running (previous call)")

# %%
## Create the Customer Support Agent (Consumer)
# Create a RemoteA2aAgent that connects to our Product Catalog Agent
# This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
remote_product_catalog_agent = RemoteA2aAgent(
    name = "product_catalog_agent",
    description = "Remote product catalog agent from external vendor that provides product information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card = f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("Remote Product Catalog Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")

# %%
# Now create the Customer Support Agent that uses the remote Product Catalog Agent
customer_support_agent = LlmAgent(
    model = Gemini(model = "gemini-2.5-flash-lite", retry_options = retry_config),
    name = "customer_support_agent",
    description = "A customer support assistant that helps customers with product inquiries and information.",
    instruction = """
    You are a friendly and professional customer support agent.
    When customers ask about products:
    1. Use the product_catalog_agent sub-agent to look up product information
    2. Provide clear answers about pricing, availability, and specifications
    3. If a product is out of stock, mention the expected availability
    4. Be helpful and professional!
    
    Always get product information from the product_catalog_agent before answering customer questions.
    """,
    sub_agents = [remote_product_catalog_agent], # Add the remote agent as a sub-agent!
)

print("Customer Support Agent created!")
print("   Model: gemini-2.5-flash-lite")
print("   Sub-agents: 1 (remote Product Catalog Agent via A2A)")
print("   Ready to help customers!")

# %%
## Test A2A Communication
async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between Customer Support Agent and Product Catalog Agent.
    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Customer Support Agent
    3. Support Agent communicates with Product Catalog Agent via A2A
    4. Displays the response
    
    Args:
        user_query: The question to sk the Customer Support Agent
    """
    # Setup session management (required by ADK)
    session_service = InMemorySessionService()
    
    # Session indentifiers
    app_name = "support_app"
    user_id = "demo_user"
    # Use unique session ID for each test to avoid conflicts
    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"
    
    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    # This pattern matches the deployment notebook exactly
    session = await session_service.create_session(
        app_name = app_name, user_id = user_id, session_id = session_id
    )
    
    # Create runner for the Customer Support Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        agent = customer_support_agent, app_name = app_name, session_service = session_service
    )

    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts = [types.Part(text = user_query)])
    
    # Display query
    print(f"\nCustomer: {user_query}")
    print(f"\nSupport Agent response:")
    print("-" * 60)
    
    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id = user_id, session_id = session_id, new_message = test_content
    ):
        # Print final response only (skip intermediate events)
        
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text"):
                    print(part.text)
    
    print("-" * 60)
    
# Run the test
print("Testing A2A Communication...\n")
await test_a2a_communication("Can you tell me about the iPhone 15 Pro? Is it in stock?")

# %%
# Tests comparing multiple products
await test_a2a_communication(
    "I'm looking for a laptop. Can you compare the Dell XPS 15 and MacBook Pro 14 for me?"
)

# %% 
# Test specific product inquiry
await test_a2a_communication(
    "Do you have the Sony WH-1000XM5 headphones? What's the price?"
)