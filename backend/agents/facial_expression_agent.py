from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agents.tools.facial_expression_tool import analyze_facial_expressions as facial_expression_tool
from dotenv import load_dotenv
import os

load_dotenv()

facial_expression_agent = Agent(
    name="facial-expression-agent",
    model=OpenAIChat(
        id="llama-3.1-8b-instant",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
    response_format={"type": "json_object"},
    tools=[facial_expression_tool],
    description="Agent that analyzes facial expressions in videos",
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)