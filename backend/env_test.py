print("Environment test: OK")
import os
print(f"Current directory: {os.getcwd()}")
import agno
print(f"Agno version: {agno.__version__ if hasattr(agno, '__version__') else 'unknown'}")
