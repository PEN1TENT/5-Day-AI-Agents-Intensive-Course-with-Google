# %%
import os
from dotenv import load_dotenv
from pygments.token import Number

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("Gemini API key setup complete.")

# %%
import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("ADK components imported successfully.")

# %%
retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504],
)

# %%
## Model Context Protocol (MCP)
# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params = StdioConnectionParams(
        server_params = StdioServerParameters(
            command = "npx",
            args = [
                "-y",
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter = ["getTinyImage"],
        ),
        timeout = 30,
    )
)

print("MCP Tool created")

# %%
# Create image agent with MCP integration
image_agent = LlmAgent(
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    name = "image_agent",
    instruction = "Use the MCP Tool to generate images for user queries",
    tools = [mcp_image_server],
)

# %%
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(agent = image_agent)

# %%
response = await runner.run_debug("Provide a sample tiny image", verbose = True)

# %%
from IPython.display import display, Image as IPImage
import base64

for event in response:
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "function_response") and part.function_response:
                for item in part.function_response.response.get("content", []):
                    if item.get("type") == "image":
                        display(IPImage(data = base64.b64decode(item["data"])))

# %%
## Kaggle MCP Server - For dataset and notebook operations
# McpToolset(
#     connection_params=StdioConnectionParams(
#         server_params=StdioServerParameters(
#             command='npx',
#             args=[
#                 '-y',
#                 'mcp-remote',
#                 'https://www.kaggle.com/mcp'
#             ],
#         ),
#         timeout=30,
#     )
# )
# 
## GitHub MCP Server - For PR/Issue analysis
# McpToolset(
#     connection_params=StreamableHTTPServerParams(
#         url="https://api.githubcopilot.com/mcp/",
#         headers={
#             "Authorization": f"Bearer {GITHUB_TOKEN}",
#             "X-MCP-Toolsets": "all",
#             "X-MCP-Readonly": "true"
#         },
#     ),
# )

# %%
## Long-Running Operations (LRO) - (Human-in-the-Loop)
# The Shipping Tool with Approval Logic
LARGE_ORDER_THRESHOLD = 5

def place_shipping_order(
    num_containers: int, destination: str, tool_context: ToolContext
) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 contaiers (LARGE_ORDER_THRESHOLD).
        Args:
            num_containers: Number of containers to ship
            destination: Shipping destination
            
        Returns:
            Dictionary with order status
    """
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: Small orders (≤5 containers) auto-approve
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: This is the first time this tool is called. Large orders need human approval - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint = f"Large order: {num_containers} containers to {destination}. Do you want to approve?",
            payload = {"num_containers": num_containers, "destination": destination},  
        )
        return {
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.
        if tool_context.tool_confirmation.confirmed:
            return {
                "status": "approved",
                "order_id": f"ORD-{num_containers}-HUMAN",
                "num_containers": num_containers,
                "destination": destination,
                "message": f"Order approved: {num_containers} containers to {destination}",
            }

print("Long-running functions created!")

# %%
# Create shipping agent with pausable tool
shipping_agent = LlmAgent(
    name = "shipping_agent",
    model = Gemini(
        model = "gemini-2.5-flash-lite", 
        retry_options = retry_config
    ),
    instruction = """You are a shipping coordinator assistant.
    When users request to ship containers:
        1. Use the place_shipping_order tool with the number of containers and destination
        2. If the order status is 'pending', inform the user that approval is requied
        3. After receiving the final result, provide a clear summary including:
            - Order status (approved/rejected)
            - Order ID (if available)
            - Nีumber of containers and destination
        4. Keep responses concise but informative
    """,
    tools = [FunctionTool(func = place_shipping_order)],
)

print("Shipping Agent created!")

# %%
# Wrap the agent in a resumable app - THIS IS THE KEY FOR LONG-RUNNING OPERATOIONS!
shipping_app = App(
    name = "shipping_coordinator",
    root_agent = shipping_agent,
    resumability_config = ResumabilityConfig(is_resumable = True),
)

print("Resumable app created!")

# %%
session_service = InMemorySessionService()

# Create runner with the resumable app
shipping_runner = Runner(
    app = shipping_app,
    session_service = session_service,
)

print("Runner created!")

# %%
## Building the Workflow
def check_for_approval(events):
    """Check if events contain an approval request.
    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

# %%
def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

# %%
def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id = approval_info["approval_id"],
        name = "adk_request_confirmation",
        response = {"confirmed": approved},
    )
    return types.Content(
        role = "user", parts = [types.Part(function_response = confirmation_response)]
    )
    
print("Helper function defined")

# %%
async def run_shipping_workflow(query: str, auto_approve: bool = True):
    """Runs a shipping workflow with approval handling.
    Args:
        query: User's shipping request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """
    
    print(f"\n{'='*60}")
    print(f"User > {query}\n")
    
    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"
    
    # Create session
    await session_service.create_session(
    app_name = "shipping_coordinator", user_id = "test_user", session_id = session_id
    )
    query_content = types.Content(role = "user", parts = [types.Part(text = query)])
    events = []
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 1: Send initial request to the Agent. If num_containers > 5, the Agent returns the special `adk_request_confirmation` event
    async for event in shipping_runner.run_async(
        user_id = "test_user", session_id = session_id, new_message = query_content
    ):
        events.append(event)
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
        approval_info = check_for_approval(events)
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW
        if approval_info:
            print(f"Pausing for approval...")
            print(f"Human Decision: {'APPROVE' if auto_approve else 'REJECT'}\n")
            
            # PATH A: Resume the agent by calling run_async() again with the approval decision
            async for event in shipping_runner.run_async(
                user_id = "test_user",
                session_id = session_id,
                new_message = create_approval_response(
                    approval_info, auto_approve
                ), # Send human decision here
                invocation_id = approval_info[
                    "invocation_id"
                ], # Critical: same invocation_id tells ADK to RESUME
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Agent > {part.text}")
        
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    else : 
        # PATH B: If the `adk_request_confirmation` is not present - no approval needed - order completed immediately.
        print_agent_response(events)
    
    print(f"{'='*60}\n")
    
print("Workflow function ready")

# %%
# Demo 1: It's a small order. Agent receives auto-approved status from tool
await run_shipping_workflow("Ship 3 containers to Singapore")

# Demo 2: Workflow simulates human decision: APPROVE
await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve = True)

# Deme 3: Workflow simulates human decision: REJECT
await run_shipping_workflow("Ship 8 containers to Los Angeles", auto_approve = False)

# %%
# Exercise: Build an Image Generation Agent with Cost Approval¶
# The scenario:

# Build an agent that generates images using the MCP server, but requires approval for "bulk" image generation:

# Single image request (1 image): Auto-approve, generate immediately
# Bulk request (>1 image): Pause and ask for approval before generating multiple images
# Explore different publicly available Image Generation MCP Servers