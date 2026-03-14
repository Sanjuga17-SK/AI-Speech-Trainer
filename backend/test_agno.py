from agno.agent import Agent
from agno.models.openai import OpenAIChat
import os
import inspect

os.environ["OPENAI_API_KEY"] = "dummy"

print("--- Agent Info ---")
print(f"Agent __init__ args: {inspect.getfullargspec(Agent.__init__).args}")
print(f"Agent __init__ kwonlyargs: {inspect.getfullargspec(Agent.__init__).kwonlyargs}")

print("\n--- OpenAIChat Info ---")
print(f"OpenAIChat __init__ args: {inspect.getfullargspec(OpenAIChat.__init__).args}")
print(f"OpenAIChat __init__ kwonlyargs: {inspect.getfullargspec(OpenAIChat.__init__).kwonlyargs}")

print("\n--- Testing Agent ---")
try:
    a = Agent(name="test")
    print("Agent(name='test') works")
except Exception as e:
    print(f"Agent(name='test') failed: {e}")

try:
    a = Agent(response_format={"type": "json_object"})
    print("Agent(response_format=...) works")
except Exception as e:
    print(f"Agent(response_format=...) failed: {e}")

print("\n--- Testing OpenAIChat ---")
try:
    m = OpenAIChat(id="gpt-4o")
    print("OpenAIChat(id='gpt-4o') works")
except Exception as e:
    print(f"OpenAIChat(id='gpt-4o') failed: {e}")

try:
    m = OpenAIChat(response_format={"type": "json_object"})
    print("OpenAIChat(response_format=...) works")
except Exception as e:
    print(f"OpenAIChat(response_format=...) failed: {e}")
