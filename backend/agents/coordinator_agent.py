from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
import os

# Simplifed Formatter Agent (70B) for Final Synthesis
formatter_agent = Agent(
    name="formatter-agent",
    model=OpenAIChat(
        id="llama-3.3-70b-versatile",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
    description="You are a data transformation agent. You synthesize raw speech data into a coaching report.",
    instructions=[
        "Base your response ONLY on the provided tool data.",
        "Ensure the 'transcription' key contains the FULL verbatim text provided in the VOICE DATA.",
        "Apply your expertise to generate scores and coaching feedback."
    ]
)

# Keep coordinator_agent as a placeholder for backward compatibility if needed, 
# but the logic is now in main.py
coordinator_agent = formatter_agent