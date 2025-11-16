# %%
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("Gemini API key setup complete.")

# %%
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
print("ADK components imported successfully.")

# %%
retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504],
)

# %%
## LLM Orchestrator
# Research Agent: Its job is to use the google_search tool and present findings.
research_agent = Agent(
    name = "ResearchAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """You are a specialized research agent. 
    Your only job is to use the google_search tool to find 2-3 pieces of relavant information on the given topic and present the findings with citations.
    """,
    tools = [google_search],
    output_key = "research_findings",
)

print("research_agent created.")

# %%
# Summarizer Agent: Its job is to summarize the text it receives.
research_findings = ""
summarizer_agent = Agent(
    name = "SummarizerAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""Read the provided research findings: {research_findings}
    Create a concise summary as a bulleted list with 3-5 key points.
    """,
    output_key = "final_summary",
)

print("summarizer_agent created.")

# %%
# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
root_agent = Agent(
    name = "ResearchCoordinator",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
    1. First, you MUST call the `ResearchAgent` tool to find relavant information on the topic provided by the user.
    2. Next, after receiving the research findings, you Must call the `SummarizerAgent` tool to create a concise summary.
    3. Finally, present the final summary clearly to the user as your response.""",
    tools = [AgentTool(research_agent), AgentTool(summarizer_agent)],
)
print("root_agent created.")
# %%
runner = InMemoryRunner(agent = root_agent)
response = await runner.run_debug(
    "What are the latest advancements in quantum computing and what do they mean for AI?"
)

# %%
## Sequential Workflows - The Assembly Line
# Outline Agent: Creates the initial blog post outline.
outline_agent = Agent(
    name = "OutlineAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """Create a blog outline for the given topic with:
        1. A catchy headline
        2. An introduction hook
        3. 3-5 main sections with 2-3 bullet points for each
        4. A concluding thought""",
        output_key = "blog_outline",
)

print("outline_agent created.")

# %%
# Writer Agent: Writes the full blog post based on the outline from the previous agent.
blog_outline = ""
writer_agent = Agent(
    name = "WriterAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key = "blog_draft",
)

print("writer_agent created.")

# %%
# Editor Agent: Edits and polishes the draft from the writer agent.
blog_draft = ""
editor_agent = Agent(
    name = "EditorAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""Edit this draft: {blog_draft}
    Your task is to pulish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key = "final_blog",
)

print("editor_agent created.")

# %%
root_agent = SequentialAgent(
    name = "BlogPipeline",
    sub_agents = [outline_agent, writer_agent, editor_agent],
)

print("Sequential Agent created.")

# %%
runner = InMemoryRunner(agent = root_agent)
response = await runner.run_debug(
    "Write a blog post about the benefits of multi-agent systems for software developers"
)

# %%
## Parallel Workflows - Independent Researchers
# Tech Researcher: Focusess on AI and ML trends.
tech_researcher = Agent(
    name = "TechResearcher",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """Research the latest AI/ML trends. 
    Include 3 key developments, the main companies involved, and the potential impact.
    Keep the report very concise (100 words).""",
    tools = [google_search],
    output_key = "tech_research",
)

print("tech_researcher created.")

# %%
# Health Researcher: Focuses on medical breakthroughs.
health_researcher = Agent(
    name = "HealthResearcher",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """Research recent medical breakthroughs. 
    Include 3 significant advances, their practical applications, and estimated timelines.
    Keep the report concise (100 words).""",
    tools = [google_search],
    output_key = "health_research",
)

print("health_researcher created.")

# %%
# Finance Researcher: Focuses on fintech trends.
finance_researcher = Agent(
    name = "FinanceResearcher",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """Research current fintech trends. Include 3 key trends,
    their market implications, and the future outlook. 
    Keep the report concise (100 words).""",
    tools = [google_search],
    output_key = "finance_research",
)

print("finance_researcher created.")

# %%
# The AggregatorAgent runs *after* the parallel step to synthesize the results.
tech_research = ""
health_research = ""
finance_research = ""
aggregator_agent = Agent(
    name = "AggregatorAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""Combine these three research findings into a single executive summary:
        **Technology Trends:**
        {tech_research}
        
        **Health Breakthroughs:
        {health_research}
        
        **Finance Innovations:
        {finance_research}
        
        Your summary should highlight common themes, surprising connections, 
        and the most important key takeaways form all three reports. 
        The final summary should be around 200 words.""",
        output_key = "executive_summary",
) 

print("aggregator_agent created.")

# %%
# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name = "ParallelResearchTeam",
    sub_agents = [tech_researcher, health_researcher, finance_researcher],
)

# This SeuentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name = "ResearchSystem",
    sub_agents = [parallel_research_team, aggregator_agent],
)

print("Parallel and Sequential Agents created.")

# %%
runner = InMemoryRunner(agent = root_agent)
response = await runner.run_debug(
    "Run the daily executive briefing on Tech, Health, and Finance"
)

# %%
## Loop Workflows - The Refinement Cycle
# This agent runs ONCE at the beginning to create the first fradt.
initial_writer_agent = Agent(
    name = "InitialWriterAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """Based on the user's prompt, 
    write the first draft of a short story (around 100-150 words).
    Output only the story text, with no instruction or explanation.""",
    output_key = "current_story",
)

print("initial_writer_agent created.")

# %%
# This agent's only job is to provide feedback or the approval signal. It has no tools.
current_story = ""
critic_agent = Agent(
    name = "CriticAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""You are a constructive story critic.
    Reciew the story provided below.
    Story: {current_story}
    
    Evaluate the story's plot, characters, and pacing.
    -If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
    -Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key = "critique",
)

print("critic_agent created.")

# %%
# This is the function that the RefinerAgent will call to exit the loop.
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED',
    indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}
    
print("exit_loop function created.")

# %%
# This agent refines the story based on critique OR calls the exit_loop function.
critique = ""
refiner_agent = Agent(
    name = "RefinerAgent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = f"""You are a story refiner. You have a story draft and critique.
    
    Story Draft: {current_story}
    Critique: {critique}
    
    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
    output_key = "current_story", 
    tools = [FunctionTool(exit_loop)],
) 

print("refiner_agent created.")

# %%
# This LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
story_refinement_loop = LoopAgent(
    name = "StoryRefinementLoop",
    sub_agents = [critic_agent, refiner_agent],
    max_iterations = 2,
)

# The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
root_agent = SequentialAgent(
    name = "StoryPipeline",
    sub_agents = [initial_writer_agent, story_refinement_loop]
)

print("Loop and Sequential Agents created.")

# %%
runner = InMemoryRunner(agent = root_agent)
response = await runner.run_debug(
    "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map"
)


