# %%
import os

from dotenv import load_dotenv
from pydantic.types import NonNegativeFloat
from pygments.token import Number
from requests.sessions import session
from sqlalchemy.sql.coercions import TruncatedLabelImpl

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("Gemini API key setup complete.")

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
!adk create home_automation_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

# %%
url_prefix = get_adk_proxy_url()

# %%
!adk web --url_prefix {url_prefix}

# %%
## Systematic Evaluation
import json

# Create evaluation configuration with basic criteria
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 1.0,  # Perfect tool usage required
        "response_match_score": 0.8,  # 80% text similarity threshold
    }
}

with open("home_automation_agent/test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print("Evaluation configuration created!")
print("\nEvaluation Criteria:")
print("- tool_trajectory_avg_score: 1.0 - Requires exact tool usage match")
print("- response_match_score: 0.8 - Requires 80% text similarity")
print("\nWhat this evaluation will catch:")
print("Incorrect tool usage (wrong device, location, or status)")
print("Poor response quality and communication")
print("Deviations from expected behavior patterns")
# %%
test_cases = {
    "eval_set_id": "home_automation_integration_suite",
    "eval_cases": [
        {
            "eval_id": "living_room_light_on",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Please turn on the floor lamp in the living room"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "Successfully set the floor lamp in the living room to on."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "set_device_status",
                                "args": {
                                    "location": "living room",
                                    "device_id": "floor lamp",
                                    "status": "ON",
                                },
                            }
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "kitchen_on_off_sequence",
            "conversation": [
                {
                    "user_content": {
                        "parts": [{"text": "Switch on the main light in the kitchen."}]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "Successfully set the main light in the kitchen to on."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "set_device_status",
                                "args": {
                                    "location": "kitchen",
                                    "device_id": "main light",
                                    "status": "ON",
                                },
                            }
                        ]
                    },
                }
            ],
        },
    ],
}

# %%
import json

with open("/Users/thumma/coding/5-Day AI Agents /Implementation/integration.evalset.json", "w") as f: json.dump(test_cases, f, indent = 2)
    
print("Evaluation test cases created")
print("\n Test scenarios:")
for case in test_cases["eval_cases"]:
    user_msg = case["conversation"][0]["user_content"]["parts"][0]["text"]
    print(f"- {case['eval_id']}: {user_msg}")
    
    print("\nExpected results:")
    print("- basic_device_control: Should pass both criteria")
    print("- wrong_tool_usage_test: May fail tool_trajectory if agent uses wrong parameters")
    print("- poor_response_quality_test: May fail response_match if response differs too m")
# %% 
print("Run this command to execute evaluation:")
!adk eval home_automation_agent home_automation_agent/integration.evalset.json --config_file_path home_automation_agent/test_config.json --print_detailed_results

# %%
# Analyzing evaluation results - the data science approach
print("Understanding Evaluation Results:")
print()
print("EXAMPLE ANALYSIS:")
print()
print("Test Case: living_room_light_on")
print("   response_match_score: 0.45/0.80")
print("   ool_trajectory_avg_score: 1.0/1.0")
print()
print("What this tells us:")
print("- TOOL USAGE: Perfect - Agent used correct tool with correct parameters")
print("- RESPONSE QUALITY: Poor - Response text too different from expected")
print("- ROOT CAUSE: Agent's communication style, not functionality")
print()
print("ACTIONABLE INSIGHTS:")
print("1. Technical capability works (tool usage perfect)")
print("2. Communication needs improvement (response quality failed)")
print("3. Fix: Update agent instructions for clearer language or constrained response.")
print()
